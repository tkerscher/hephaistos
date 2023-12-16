#include "hephaistos/command.hpp"

#include <algorithm>
#include <sstream>
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

SubroutineBuilder& SubroutineBuilder::addCommand(const Command& command) & {
    if (!*this)
        throw std::runtime_error("SubroutineBuilder has already finished!");

    command.record(*cmdBuffer);
    return *this;
}
SubroutineBuilder SubroutineBuilder::addCommand(const Command& command) && {
    static_cast<SubroutineBuilder&>(*this).addCommand(command);
    return std::move(*this);
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

uint64_t Timeline::getId() const {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(timeline->semaphore));
}

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

bool Submission::forgettable() const noexcept {
    return !resources || resources->commands.empty();
}

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
    if (!forgettable()) {
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
    //for recording commands
    VkCommandPool pool;
    vulkan::Command recordingCmd = {};
    std::vector<VkCommandBuffer> recordedBuffers = {};

    void finishRecording() {
        //any buffer to finish?
        if (!recordingCmd.buffer)
            return;

        //end recording
        vulkan::checkResult(context.fnTable.vkEndCommandBuffer(
            recordingCmd.buffer));
        commandBuffers.push_back(recordingCmd.buffer);
        recordedBuffers.push_back(recordingCmd.buffer);
        waitStages.back() |= recordingCmd.stage;
        //reset recording command buffer
        recordingCmd = {};
    }

    std::vector<VkPipelineStageFlags> waitStages = {};
    std::vector<VkCommandBuffer> commandBuffers = {};

    std::vector<uint64_t> waitValues = {};
    std::vector<VkSemaphore> waitSemaphores = {};
    std::vector<uint64_t> signalValues = {};
    std::vector<VkSemaphore> signalSemaphores = {};
    std::vector<VkSubmitInfo> submitInfos = {};
    std::vector<VkTimelineSemaphoreSubmitInfo> timelineInfos = {};
    //the value to wait for in the next batch
    uint64_t currentValue = 0;

    //Handle to manage lifetime of implicit timeline
    std::unique_ptr<Timeline> exclusiveTimeline;
    Timeline& timeline;
    VkSemaphore semaphore; //timeline semaphore

    const vulkan::Context& context;

    pImp(Timeline& timeline, uint64_t value)
        : pool(fetchCommandPool(*timeline.getContext()))
        , currentValue(value)
        , exclusiveTimeline(nullptr)
        , timeline(timeline)
        , semaphore(timeline.getTimeline().semaphore)
        , context(*timeline.getContext())
    {}

    pImp(ContextHandle context)
        : pool(fetchCommandPool(*context))
        , exclusiveTimeline(std::make_unique<Timeline>(std::move(context)))
        , timeline(*exclusiveTimeline)
        , semaphore(exclusiveTimeline->getTimeline().semaphore)
        , context(*timeline.getContext())
    {}
};

SequenceBuilder::operator bool() const {
    return static_cast<bool>(_pImp);
}

SequenceBuilder& SequenceBuilder::And(const Command& command) & {
    if (!*this)
        throw std::runtime_error("SequenceBuilder has already finished!");

    //Check if we already started a command buffer
    if (!_pImp->recordingCmd.buffer) {
        //allocate and start a new command buffer
        VkCommandBufferAllocateInfo allocInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = _pImp->pool,
            .commandBufferCount = 1
        };
        vulkan::checkResult(_pImp->context.fnTable.vkAllocateCommandBuffers(
            _pImp->context.device, &allocInfo, &_pImp->recordingCmd.buffer));
        //start recording
        vulkan::checkResult(_pImp->context.fnTable.vkBeginCommandBuffer(
            _pImp->recordingCmd.buffer, &BeginInfo));

        //mark submission
        _pImp->submitInfos.back().commandBufferCount += 1;
    }

    //Record command
    command.record(_pImp->recordingCmd);

    return *this;
}

SequenceBuilder& SequenceBuilder::And(const Subroutine& subroutine) & {
    if (!*this)
        throw std::runtime_error("SequenceBuilder has already finished!");

    //add command buffer
    auto& cmd = subroutine.getCommandBuffer();
    _pImp->commandBuffers.push_back(cmd.buffer);
    _pImp->waitStages.back() |= cmd.stage;
    _pImp->submitInfos.back().commandBufferCount += 1;

    return *this;
}

SequenceBuilder& SequenceBuilder::NextStep() & {
    if (!*this)
        throw std::runtime_error("SequenceBuilder has already finished!");

    //finish previous command buffer
    _pImp->finishRecording();

    //create new submission
    _pImp->timelineInfos.push_back(VkTimelineSemaphoreSubmitInfo{
        .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
        .waitSemaphoreValueCount = 1,
        .signalSemaphoreValueCount = 1
    });
    _pImp->submitInfos.push_back(VkSubmitInfo{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = 1,
        //we only ever signal our own timeline
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &_pImp->semaphore
    });

    //save wait values & semaphores
    _pImp->waitValues.push_back(_pImp->currentValue);
    _pImp->waitSemaphores.push_back(_pImp->semaphore);
    _pImp->currentValue += 1;
    _pImp->signalValues.push_back(_pImp->currentValue);
    _pImp->signalSemaphores.push_back(_pImp->semaphore);
    _pImp->waitStages.push_back(0);

    return *this;
}

SequenceBuilder& SequenceBuilder::Then(const Command& command) & {
    NextStep();
    And(command);
    return *this;
}

SequenceBuilder& SequenceBuilder::Then(const Subroutine& subroutine) & {
    NextStep();
    And(subroutine);
    return *this;
}

SequenceBuilder& SequenceBuilder::WaitFor(uint64_t value) & {
    if (!*this)
        throw std::runtime_error("SequenceBuilder has already finished!");

    //wait for will dead lock if we use an implicit timeline
    if (_pImp->exclusiveTimeline)
        throw std::logic_error("WaitFor will dead lock with an implicit timeline!");

    //check if we have an open submission
    if (_pImp->submitInfos.back().commandBufferCount > 0)
        NextStep();

    //update wait/signal values
    //they are always the last ones
    _pImp->waitValues.back() = value;
    _pImp->signalValues.back() = value + 1;
    _pImp->currentValue = value + 1;

    return *this;
}

SequenceBuilder& SequenceBuilder::WaitFor(const Timeline& timeline, uint64_t value) & {
    if (!*this)
        throw std::runtime_error("SequenceBuilder has already finished!");

    //check if timeline is the internal one
    if (timeline.getTimeline().semaphore == _pImp->semaphore)
        return WaitFor(value);

    //check if we have an open submission
    if (_pImp->submitInfos.back().commandBufferCount > 0)
        NextStep();

    _pImp->submitInfos.back().waitSemaphoreCount += 1;
    _pImp->timelineInfos.back().waitSemaphoreValueCount += 1;
    //ensure the last element in wait values/semaphores is the signaling timeline:
    // ... | back() | => ... | value | back() |
    // i.e. insert in second to last place
    //This is so WaitFor(uint64_t) wait works by simply alter the last value
    auto semaphore = timeline.getTimeline().semaphore;
    _pImp->waitSemaphores.insert(_pImp->waitSemaphores.end() - 1, semaphore);
    _pImp->waitValues.insert(_pImp->waitValues.end() - 1, value);

    return *this;
}

SequenceBuilder SequenceBuilder::And(const Command& command) && {
    static_cast<SequenceBuilder&>(*this).And(command);
    return std::move(*this);
}
SequenceBuilder SequenceBuilder::And(const Subroutine& subroutine) && {
    static_cast<SequenceBuilder&>(*this).And(subroutine);
    return std::move(*this);
}
SequenceBuilder SequenceBuilder::NextStep() && {
    static_cast<SequenceBuilder&>(*this).NextStep();
    return std::move(*this);
}
SequenceBuilder SequenceBuilder::Then(const Command& command) && {
    static_cast<SequenceBuilder&>(*this).Then(command);
    return std::move(*this);
}
SequenceBuilder SequenceBuilder::Then(const Subroutine& subroutine) && {
    static_cast<SequenceBuilder&>(*this).Then(subroutine);
    return std::move(*this);
}
SequenceBuilder SequenceBuilder::WaitFor(uint64_t value) && {
    static_cast<SequenceBuilder&>(*this).WaitFor(value);
    return std::move(*this);
}
SequenceBuilder SequenceBuilder::WaitFor(const Timeline& timeline, uint64_t value) && {
    static_cast<SequenceBuilder&>(*this).WaitFor(timeline, value);
    return std::move(*this);
}

Submission SequenceBuilder::Submit() {
    if (!*this)
        throw std::runtime_error("SequenceBuilder has already finished!");

    //finish previous command buffer
    _pImp->finishRecording();

    //update submit infos
    //since the vectors may change their memory location as they grow
    //we can only now fill in the pointers in the submission infos

    //also duplicate waitStages to match size of semaphores
    auto semaphoreCount = _pImp->waitValues.size();
    std::vector<VkPipelineStageFlags> waitStages(semaphoreCount);

    auto pWaitStage = waitStages.data();
    auto pWaitValue = _pImp->waitValues.data();
    auto pWaitSemaphore = _pImp->waitSemaphores.data();
    auto pSignalValue = _pImp->signalValues.data();
    auto pCmd = _pImp->commandBuffers.data();

    auto pTimeline = _pImp->timelineInfos.data();
    auto pSubmit = _pImp->submitInfos.data();

    auto submitCount = _pImp->submitInfos.size();
    for (auto i = 0u; i < submitCount; ++i, ++pTimeline, ++pSubmit) {
        //edge case: empty submission can't have waitStage=0
        //  -> use TOP_OF_PIPE
        if (_pImp->waitStages[i] == 0)
            _pImp->waitStages[i] = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        //duplicate wait stages
        std::fill_n(pWaitStage, pSubmit->waitSemaphoreCount, _pImp->waitStages[i]);

        //fix timeline info
        pTimeline->pSignalSemaphoreValues = pSignalValue;
        pTimeline->pWaitSemaphoreValues = pWaitValue;
        pSignalValue += pTimeline->signalSemaphoreValueCount;
        pWaitValue += pTimeline->waitSemaphoreValueCount;

        //fix submit info
        pSubmit->pNext = pTimeline;
        pSubmit->pCommandBuffers = pCmd;
        pSubmit->pWaitSemaphores = pWaitSemaphore;
        pSubmit->pWaitDstStageMask = pWaitStage;
        pCmd += pSubmit->commandBufferCount;
        pWaitSemaphore += pSubmit->waitSemaphoreCount;
        pWaitStage += pSubmit->waitSemaphoreCount;
    }

    //submit
    vulkan::checkResult(_pImp->context.fnTable.vkQueueSubmit(
        _pImp->context.queue, submitCount, _pImp->submitInfos.data(), nullptr));

    //if there are no recorded command buffers we can already give the pool back
    if (_pImp->recordedBuffers.empty()) {
        _pImp->context.sequencePool.push(_pImp->pool);
        _pImp->pool = VK_NULL_HANDLE;
    }

    //prepare submission
    auto resource = std::unique_ptr<SubmissionResources>(
        new SubmissionResources{
            _pImp->pool,
            std::move(_pImp->recordedBuffers),
            std::move(_pImp->exclusiveTimeline)
        }
    );
    auto& timeline = _pImp->timeline;
    auto finalStep = _pImp->currentValue;
    //free _pImp to prevent multiple submission
    _pImp.reset();
    //build and return submission responsible for the resources lifetime
    return Submission{
        timeline, finalStep, std::move(resource)
    };
}

std::string SequenceBuilder::printWaitGraph() const {
    if (!*this)
        throw std::runtime_error("SequenceBuilder has already finished!");

    std::stringstream out;
    auto n = _pImp->submitInfos.size();
    auto pSub = _pImp->submitInfos.data();
    auto pWait = _pImp->waitValues.data();
    auto pWaitSem = _pImp->waitSemaphores.data();
    auto pSig = _pImp->signalValues.data();
    for (auto i = 0u; i < n; ++i) {
        // print wait values
        auto nWaits = _pImp->timelineInfos[i].waitSemaphoreValueCount;
        for (auto ii = 0u; ii < nWaits; ++ii) {
            out << *(pWaitSem++) << '(' << *(pWait++) << ") ";
        }
        // print #commands/subroutines
        out << "-> (" << (pSub++)->commandBufferCount << ") -> ";
        // print signal values
        out << _pImp->semaphore << '(' << *(pSig++) << ")\n";
    }
    return out.str();
}

SequenceBuilder::SequenceBuilder(SequenceBuilder&& other) noexcept = default;
SequenceBuilder& SequenceBuilder::operator=(SequenceBuilder&& other) noexcept = default;

SequenceBuilder::SequenceBuilder(Timeline& timeline, uint64_t startValue)
    : _pImp(new pImp(timeline, startValue))
{
    //init pImp
    NextStep();
}
SequenceBuilder::SequenceBuilder(ContextHandle context)
    : _pImp(new pImp(context))
{
    //init pImp
    NextStep();
}
SequenceBuilder::~SequenceBuilder() {
    if (_pImp) {
        //free recorded command buffers
        if (!_pImp->recordedBuffers.empty()) {
            _pImp->context.fnTable.vkFreeCommandBuffers(
                _pImp->context.device, _pImp->pool,
                static_cast<uint32_t>(_pImp->recordedBuffers.size()),
                _pImp->recordedBuffers.data());
        }
        //free currently recording command buffer
        if (_pImp->recordingCmd.buffer) {
            _pImp->context.fnTable.vkFreeCommandBuffers(
                _pImp->context.device, _pImp->pool,
                1, &_pImp->recordingCmd.buffer);
        }
        //reset pool
        vulkan::checkResult(_pImp->context.fnTable.vkResetCommandPool(
            _pImp->context.device, _pImp->pool,
            VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT));
        //return pool to context
        _pImp->context.sequencePool.push(_pImp->pool);
    }
}

void execute(const ContextHandle& context, const Command& command) {
    //run a one time submit command
    vulkan::oneTimeSubmit(*context, [&command](VkCommandBuffer cmd) {
        vulkan::Command wrapper{ cmd, 0 };
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
