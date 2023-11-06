#include "hephaistos/command.hpp"

#include <vector>

#include "volk.h"

#include "vk/result.hpp"
#include "vk/types.hpp"
#include "vk/util.hpp"

namespace hephaistos {

Command::~Command() = default;

/******************************** SUBROUTINE **********************************/

bool Subroutine::simultaneousUse() const {
    return simultaneous_use;
}
const vulkan::Command& Subroutine::getCommandBuffer() const {
    return *cmdBuffer;
}

Subroutine::Subroutine(Subroutine&&) noexcept = default;
Subroutine& Subroutine::operator=(Subroutine&&) noexcept = default;

Subroutine::Subroutine(
    ContextHandle context,
    std::unique_ptr<vulkan::Command> cmdBuffer,
    bool simultaneous_use)
    : Resource(std::move(context))
    , cmdBuffer(std::move(cmdBuffer))
    , simultaneous_use(simultaneous_use)
{}
Subroutine::~Subroutine() {
    if (cmdBuffer) {
        auto& context = *getContext();
        context.fnTable.vkFreeCommandBuffers(
            context.device, context.subroutinePool,
            1, &cmdBuffer->buffer);
    }
}

SubroutineBuilder::operator bool() const {
    return static_cast<bool>(cmdBuffer);
}

SubroutineBuilder& SubroutineBuilder::addCommand(const Command& command) {
    if (!*this)
        throw std::runtime_error("SubroutineBuilder has already finished!");

    command.record(*cmdBuffer);
    return *this;
}
Subroutine SubroutineBuilder::finish() {
    if (!*this)
        throw std::runtime_error("SubroutineBuilder has already finished!");

    //end recording & build subroutine
    vulkan::checkResult(context->fnTable.vkEndCommandBuffer(cmdBuffer->buffer));
    return Subroutine(std::move(context), std::move(cmdBuffer), simultaneous_use);
}

SubroutineBuilder::SubroutineBuilder(SubroutineBuilder&&) noexcept = default;
SubroutineBuilder& SubroutineBuilder::operator=(SubroutineBuilder&&) noexcept = default;

SubroutineBuilder::SubroutineBuilder(ContextHandle context, bool simultaneous_use)
    : context(std::move(context))
    , cmdBuffer(std::make_unique<vulkan::Command>())
    , simultaneous_use(simultaneous_use)
{
    //Allocate command buffer
    VkCommandBufferAllocateInfo allocInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = this->context->subroutinePool,
        .commandBufferCount = 1
    };
    vulkan::checkResult(this->context->fnTable.vkAllocateCommandBuffers(
        this->context->device, &allocInfo, &cmdBuffer->buffer));

    //start recording
    VkCommandBufferBeginInfo beginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
    };
    if (simultaneous_use)
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    vulkan::checkResult(this->context->fnTable.vkBeginCommandBuffer(
        cmdBuffer->buffer, &beginInfo));
}
SubroutineBuilder::~SubroutineBuilder() {
    if (cmdBuffer) {
        context->fnTable.vkFreeCommandBuffers(
            context->device, context->subroutinePool,
            1, &cmdBuffer->buffer);
    }
}

/********************************* TIMELINE ***********************************/

uint64_t Timeline::getValue() const {
    uint64_t value;
    auto& context = getContext();
    vulkan::checkResult(context->fnTable.vkGetSemaphoreCounterValue(
        context->device, timeline->semaphore, &value));
    return value;
}
void Timeline::setValue(uint64_t value) {
    auto& context = getContext();
    VkSemaphoreSignalInfo info{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
        .semaphore = timeline->semaphore,
        .value = value
    };
    vulkan::checkResult(context->fnTable.vkSignalSemaphore(
        context->device, &info));
}
void Timeline::waitValue(uint64_t value) const {
    static_cast<void>(waitValue(value, UINT64_MAX));
}
bool Timeline::waitValue(uint64_t value, uint64_t timeout) const {
    auto& context = getContext();
    VkSemaphoreWaitInfo info{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .semaphoreCount = 1,
        .pSemaphores = &timeline->semaphore,
        .pValues = &value
    };
    auto result = context->fnTable.vkWaitSemaphores(
        context->device, &info, timeout);

    switch (result) {
    case VK_SUCCESS:
        return true;
    case VK_TIMEOUT:
        return false;
    default:
        vulkan::checkResult(result); //will throw
        return false;
    }
}

vulkan::Timeline& Timeline::getTimeline() const {
    return *timeline;
}

Timeline::Timeline(Timeline&& other) noexcept
    : Resource(std::move(other))
    , timeline(std::move(other.timeline))
{}
Timeline& Timeline::operator=(Timeline&& other) noexcept {
    Resource::operator=(std::move(other));
    timeline = std::move(other.timeline);
    return *this;
}

Timeline::Timeline(ContextHandle context, uint64_t initialValue)
    : Resource(std::move(context))
    , timeline(std::make_unique<vulkan::Timeline>())
{
    auto& _context = getContext();

    VkSemaphoreTypeCreateInfo type{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
        .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
        .initialValue = initialValue
    };
    VkSemaphoreCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = &type
    };
    vulkan::checkResult(_context->fnTable.vkCreateSemaphore(
        _context->device, &info, nullptr, &timeline->semaphore));
}
Timeline::~Timeline() {
    if (timeline) {
        auto& context = getContext();
        context->fnTable.vkDestroySemaphore(
            context->device, timeline->semaphore, nullptr);
    }
}

/******************************** SUBMISSION *********************************/

struct SubmissionResources {
    VkCommandPool pool;
    std::vector<VkCommandBuffer> commands;
    //Handle used to manage lifetime of implicit timeline
    std::unique_ptr<Timeline> exclusiveTimeline = nullptr;
};

const Timeline& Submission::getTimeline() const { return timeline.get(); }
uint64_t Submission::getFinalStep() const { return finalStep; }

void Submission::wait() const {
    if (finalStep > 0)
        timeline.get().waitValue(finalStep);
}
bool Submission::wait(uint64_t timeout) const {
    return (finalStep == 0 || timeline.get().waitValue(finalStep, timeout));
}

Submission::Submission(Submission&& other) noexcept
    : finalStep(other.finalStep)
    , timeline(std::move(other.timeline))
    , resources(std::move(other.resources))
{
    //make waits no-op to be safe
    other.finalStep = 0;
}
Submission& Submission::operator=(Submission&& other) noexcept {
    finalStep = other.finalStep;
    timeline = std::move(other.timeline);
    resources = std::move(other.resources);
    //make waits no-op to be safe
    other.finalStep = 0;

    return *this;
}

Submission::Submission(const Timeline& timeline, uint64_t finalStep, std::unique_ptr<SubmissionResources> resources)
    : finalStep(finalStep)
    , timeline(std::cref(timeline))
    , resources(std::move(resources))
{}
Submission::~Submission() {
    //reset pool if there is one
    if (resources && resources->commands.size() > 0) {
        //ensure we're save to reset pool by waiting submission to finish
        wait();

        auto& context = timeline.get().getContext();
        //free command buffers
        context->fnTable.vkFreeCommandBuffers(
            context->device,
            resources->pool,
            static_cast<uint32_t>(resources->commands.size()),
            resources->commands.data());
        //reset pool
        vulkan::checkResult(context->fnTable.vkResetCommandPool(
            context->device,
            resources->pool,
            VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT));
        //return to context
        context->sequencePool.push(resources->pool);
    }
}

/********************************* SEQUENCE **********************************/

namespace {

//Some structs are always the same -> reuse them
constexpr VkCommandBufferBeginInfo BeginInfo{
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
};

constexpr VkPipelineStageFlags EmptyStage =
    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

VkCommandPool fetchCommandPool(const vulkan::Context& context) {
    //fetch used pool or create new
    VkCommandPool pool;
    if (context.sequencePool.empty()) {
        //create new
        VkCommandPoolCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .queueFamilyIndex = context.queueFamily
        };
        vulkan::checkResult(context.fnTable.vkCreateCommandPool(
            context.device, &info, nullptr, &pool));
    }
    else {
        pool = context.sequencePool.front();
        context.sequencePool.pop();
    }
    return pool;
}

}

struct SequenceBuilder::pImp {
    std::vector<vulkan::Command> commands;
    VkCommandPool pool;

    std::vector<vulkan::Command> subroutines;
    std::vector<uint32_t> subroutineCounts;

    std::vector<uint64_t> waitValues;
    std::vector<uint64_t> signalValues;
    //the value to wait for in the next batch. Might differ from signalValues.back()
    uint64_t currentValue;

    //Handle used to manage lifetime of implicit timeline
    std::unique_ptr<Timeline> exclusiveTimeline;
    VkSemaphore semaphore;
    Timeline& timeline;

    const vulkan::Context& context;

    pImp(Timeline& timeline, uint64_t value)
        : commands({})
        , subroutines({})
        , subroutineCounts({})
        , pool(fetchCommandPool(*timeline.getContext()))
        , exclusiveTimeline(nullptr)
        , semaphore(timeline.getTimeline().semaphore)
        , timeline(timeline)
        , currentValue(value + 1)
        , context(*timeline.getContext())
    {
        commands.push_back({});
        subroutineCounts.push_back(0);
        waitValues.push_back(value);
        signalValues.push_back(value + 1);
    }
    pImp(ContextHandle context)
        : commands({})
        , subroutines({})
        , subroutineCounts({})
        , pool(fetchCommandPool(*context))
        , exclusiveTimeline(std::make_unique<Timeline>(std::move(context)))
        , semaphore(exclusiveTimeline->getTimeline().semaphore)
        , timeline(*exclusiveTimeline)
        , currentValue(1)
        , context(*timeline.getContext())
    {
        commands.push_back({});
        subroutineCounts.push_back(0);
        waitValues.push_back(0);
        signalValues.push_back(1);
    }
};

SequenceBuilder::operator bool() const {
    return static_cast<bool>(_pImp);
}

SequenceBuilder& SequenceBuilder::And(const Command& command) {
    if (!*this)
        throw std::runtime_error("SequenceBuilder has already finished!");

    //Check if we already started a command buffer
    auto& vCmd = _pImp->commands.back();
    if (!vCmd.buffer) {
        //allocate and start a new command buffer
        VkCommandBufferAllocateInfo allocInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = _pImp->pool,
            .commandBufferCount = 1
        };
        vulkan::checkResult(_pImp->context.fnTable.vkAllocateCommandBuffers(
            _pImp->context.device, &allocInfo, &vCmd.buffer));
        //start recording
        vulkan::checkResult(_pImp->context.fnTable.vkBeginCommandBuffer(
            vCmd.buffer, &BeginInfo));
    }

    //Record command
    command.record(vCmd);

    return *this;
}

SequenceBuilder& SequenceBuilder::And(const Subroutine& subroutine) {
    if (!*this)
        throw std::runtime_error("SequenceBuilder has already finished!");

    //command buffers are just pointers/handles so they are safe to just copy
    _pImp->subroutineCounts.back()++;
    _pImp->subroutines.push_back(subroutine.getCommandBuffer());

    return *this;
}

SequenceBuilder& SequenceBuilder::NextStep() {
    if (!*this)
        throw std::runtime_error("SequenceBuilder has already finished!");

    //Finish previous command buffer
    auto& prevCmd = _pImp->commands.back();
    if (prevCmd.buffer)
        vulkan::checkResult(_pImp->context.fnTable.vkEndCommandBuffer(prevCmd.buffer));

    //add new level
    _pImp->commands.emplace_back();
    _pImp->subroutineCounts.push_back(0);
    _pImp->waitValues.push_back(_pImp->currentValue);
    _pImp->currentValue++;
    _pImp->signalValues.push_back(_pImp->currentValue);

    return *this;
}

SequenceBuilder& SequenceBuilder::Then(const Command& command) {
    NextStep();
    And(command);
    return *this;
}

SequenceBuilder& SequenceBuilder::Then(const Subroutine& subroutine) {
    NextStep();
    And(subroutine);
    return *this;
}

SequenceBuilder& SequenceBuilder::WaitFor(uint64_t value) {
    if (value < _pImp->currentValue)
        throw std::logic_error("Wait value too low! Timeline can only go forward!");

    //wait for will dead lock if we use an implicit timeline
    if (_pImp->exclusiveTimeline)
        throw std::logic_error("WaitFor will dead lock with an implicit timeline!");

    //Finish previous command buffer
    auto& prevCmd = _pImp->commands.back();
    if (prevCmd.buffer) {
        vulkan::checkResult(_pImp->context.fnTable.vkEndCommandBuffer(prevCmd.buffer));

        //add new level
        _pImp->commands.emplace_back();
        _pImp->subroutineCounts.push_back(0);
        _pImp->waitValues.push_back(value);
        _pImp->currentValue = value + 1;
        _pImp->signalValues.push_back(value + 1);
    }
    else {
        //if the current is empty, just update current values
        _pImp->waitValues.back() = value;
        _pImp->currentValue = value + 1;
        _pImp->signalValues.back() = value + 1;
    }

    return *this;
}

Submission SequenceBuilder::Submit() {
    if (!_pImp)
        throw std::logic_error("A sequence can only be submitted once!");

    //Finish previous command buffer
    auto& prevCmd = _pImp->commands.back();
    if (prevCmd.buffer)
        vulkan::checkResult(_pImp->context.fnTable.vkEndCommandBuffer(prevCmd.buffer));

    //count how much command buffers we have in total
    auto steps = _pImp->commands.size();
    std::vector<uint32_t> stepSize(steps, 0);
    auto count = 0u;
    for (auto i = 0u; i < steps; ++i) {
        //recorded any commands?
        if (_pImp->commands[i].buffer) stepSize[i]++;
        //add subroutine count
        stepSize[i] += _pImp->subroutineCounts[i];
        //add total count
        count += stepSize[i];
    }

    //collect all command buffers
    std::vector<VkCommandBuffer> cmdBuffers(count);
    std::vector<VkPipelineStageFlags> stageFlags(steps);
    auto subroutinePtr = _pImp->subroutines.begin();
    for (auto i = 0u, n = 0u; i < steps; ++i) {
        //save recorded command
        if (_pImp->commands[i].buffer) {
            cmdBuffers[n++] = _pImp->commands[i].buffer;
            stageFlags[i] |= _pImp->commands[i].stage;
        }
        //save subroutines
        for (auto j = 0u; j < _pImp->subroutineCounts[i]; ++j, ++n, ++subroutinePtr) {
            cmdBuffers[n] = subroutinePtr->buffer;
            stageFlags[i] |= subroutinePtr->stage;
        }
        //check for empty stage flags
        if (!stageFlags[0])
            stageFlags[0] = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    }

    //build timeline infos
    std::vector<VkTimelineSemaphoreSubmitInfo> timelineInfos(steps);
    auto wPtr = _pImp->waitValues.data();
    auto sPtr = _pImp->signalValues.data();
    for (auto i = 0u; i < steps; ++i, ++wPtr, ++sPtr) {
        timelineInfos[i] = {
            .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
            .waitSemaphoreValueCount = 1,
            .pWaitSemaphoreValues = wPtr,
            .signalSemaphoreValueCount = 1,
            .pSignalSemaphoreValues = sPtr
        };
    }
    //build submit infos
    std::vector<VkSubmitInfo> submits(steps);
    auto cmdPtr = cmdBuffers.data();
    auto stgPtr = stageFlags.data();
    auto tPtr = timelineInfos.data();
    for (auto i = 0u; i < steps; ++i, ++stgPtr, ++tPtr) {
        submits[i] = VkSubmitInfo{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = tPtr,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &_pImp->semaphore,
            .pWaitDstStageMask = stgPtr,
            .commandBufferCount = stepSize[i],
            .pCommandBuffers = stepSize[i] > 0 ? cmdPtr : nullptr,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &_pImp->semaphore
        };
        cmdPtr += stepSize[i];
    }

    //submit
    vulkan::checkResult(_pImp->context.fnTable.vkQueueSubmit(
        _pImp->context.queue, submits.size(), submits.data(), nullptr));

    //collect non empty recorded command buffers
    std::vector<VkCommandBuffer> recordedBuffers{};
    for (int i = 0u; i < steps; ++i) {
        if (_pImp->commands[i].buffer) {
            recordedBuffers.push_back(_pImp->commands[i].buffer);
        }
    }
    //if there are no recorded command buffers we can already give the pool back
    if (recordedBuffers.empty()) {
        auto& context = _pImp->timeline.getContext();
        context->sequencePool.push(_pImp->pool);
        _pImp->pool = VK_NULL_HANDLE;
    }

    //prepare submission
    auto resource = std::make_unique<SubmissionResources>(
        _pImp->pool,
        std::move(recordedBuffers),
        std::move(_pImp->exclusiveTimeline));
    auto& timeline = _pImp->timeline;
    auto finalStep = _pImp->currentValue;
    //free _pImp to prevent multiple submission
    _pImp.reset();
    //build and return submission responsible for the resources lifetime
    return Submission{
        timeline,
        finalStep,
        std::move(resource)
    };
}

SequenceBuilder::SequenceBuilder(SequenceBuilder&& other) noexcept = default;
SequenceBuilder& SequenceBuilder::operator=(SequenceBuilder&& other) noexcept = default;

SequenceBuilder::SequenceBuilder(Timeline& timeline, uint64_t startValue)
    : _pImp(new pImp(timeline, startValue))
{}
SequenceBuilder::SequenceBuilder(ContextHandle context)
    : _pImp(new pImp(context))
{}
SequenceBuilder::~SequenceBuilder() {
    if (_pImp) {
        auto& context = _pImp->context;
        //free command buffers
        for (auto& c : _pImp->commands) {
            context.fnTable.vkFreeCommandBuffers(
                context.device, _pImp->pool,
                1, &c.buffer);
        }
        //reset pool
        vulkan::checkResult(context.fnTable.vkResetCommandPool(
            context.device,
            _pImp->pool,
            VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT));
        //return to context
        context.sequencePool.push(_pImp->pool);
    }
}

void execute(const ContextHandle& context, const Command& command) {
    //run a one time submit command
    vulkan::oneTimeSubmit(*context, [&command](VkCommandBuffer cmd) {
        vulkan::Command wrapper { cmd, 0 };
        command.record(wrapper);
    });
}

void execute(const ContextHandle& context, const Subroutine& subroutine) {
    //submit with fence, so we can wait for it to finish
    VkSubmitInfo submitInfo{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &subroutine.getCommandBuffer().buffer
    };
    vulkan::checkResult(context->fnTable.vkQueueSubmit(
        context->queue, 1, &submitInfo, context->oneTimeSubmitFence));

    //wait for it to finish
    vulkan::checkResult(context->fnTable.vkWaitForFences(
        context->device, 1, &context->oneTimeSubmitFence, VK_TRUE, UINT64_MAX));

    //reset fence for next use
    vulkan::checkResult(context->fnTable.vkResetFences(
        context->device, 1, &context->oneTimeSubmitFence));
}

void execute(const ContextHandle& context,
    const std::function<void(vulkan::Command& cmd)>& emitter)
{
    vulkan::oneTimeSubmit(*context, [&emitter](VkCommandBuffer cmd){
        vulkan::Command wrapper{ cmd, 0 };
        emitter(wrapper);
    });
}

}
