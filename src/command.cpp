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
    return !resources ||
        (resources->commands.empty() && !resources->exclusiveTimeline);
}

bool Submission::hasFinished() const {
    return timeline.get().getValue() >= finalStep;
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
{
}
Submission::~Submission() {
    //nothing to do if it's fire and forget
    if (forgettable()) return;

    //wait on submission to finish
    wait();

    //free command buffers if hold any
    if (!resources->commands.empty()) {
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

/******************************** SUBROUTINE **********************************/

namespace vulkan {

struct Subroutine {
    Command command;

    const Context& context;
};

void destroySubroutine(Subroutine* subroutine) {
    if (!subroutine) return;

    subroutine->context.fnTable.vkFreeCommandBuffers(
        subroutine->context.device,
        subroutine->context.subroutinePool,
        1, &subroutine->command.buffer
    );

    delete subroutine;
}

SubroutineHandle beginSubroutine(const ContextHandle& context, bool simultaneous_use) {
    SubroutineHandle result{ new Subroutine({ { 0 }, *context }), destroySubroutine };

    //Allocate command buffer
    VkCommandBufferAllocateInfo allocInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = context->subroutinePool,
        .commandBufferCount = 1
    };
    vulkan::checkResult(context->fnTable.vkAllocateCommandBuffers(
        context->device, &allocInfo, &result->command.buffer));

    //start recording
    VkCommandBufferBeginInfo beginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
    };
    if (simultaneous_use)
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    vulkan::checkResult(context->fnTable.vkBeginCommandBuffer(
        result->command.buffer, &beginInfo));

    //done
    return result;
}

}

namespace {

void submitSubroutine(
    const vulkan::Subroutine& subroutine,
    const Timeline& timeline,
    uint64_t signalValue,
    std::span<const TimePoint> waitOn
) {
    VkSemaphoreSubmitInfo signalInfo{
        .sType     = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .semaphore = timeline.getTimeline().semaphore,
        .value     = signalValue,
        .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT_KHR
    };
    std::vector<VkSemaphoreSubmitInfo> waitInfo(waitOn.size());
    for (auto i = 0; i < waitOn.size(); ++i) {
        waitInfo[i] = {
            .sType     = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .semaphore = waitOn[i].timeline.get().getTimeline().semaphore,
            .value     = waitOn[i].value,
            .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT_KHR
        };
    }

    VkCommandBufferSubmitInfo cmdInfo{
        .sType         = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .commandBuffer = subroutine.command.buffer
    };

    VkSubmitInfo2 info{
        .sType                    = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .waitSemaphoreInfoCount   = static_cast<uint32_t>(waitOn.size()),
        .pWaitSemaphoreInfos      = waitOn.size() > 0 ? waitInfo.data() : nullptr,
        .commandBufferInfoCount   = 1,
        .pCommandBufferInfos      = &cmdInfo,
        .signalSemaphoreInfoCount = 1,
        .pSignalSemaphoreInfos    = &signalInfo
    };
    vulkan::checkResult(subroutine.context.fnTable.vkQueueSubmit2(
        subroutine.context.queue, 1, &info, nullptr
    ));
}

}

bool Subroutine::simultaneousUse() const {
    return simultaneous_use;
}
const vulkan::Command& Subroutine::getCommandBuffer() const {
    return subroutine->command;
}

Submission Subroutine::submit(const Timeline& timeline, uint64_t signalValue, std::span<const TimePoint> waitOn) const {
    submitSubroutine(*subroutine, timeline, signalValue, waitOn);  
    return Submission{ timeline, signalValue, {} };
}
Submission Subroutine::submit(const Timeline& timeline, uint64_t signalValue) const {
    return submit(timeline, signalValue, {});
}
Submission Subroutine::submit(std::span<const TimePoint> waitOn) const {
    auto exclusiveTimeline = std::make_unique<Timeline>(getContext());
    auto& timeline = *exclusiveTimeline;
    submitSubroutine(*subroutine, timeline, 1, waitOn);
    auto resource = std::unique_ptr<SubmissionResources>(
        new SubmissionResources{ nullptr, {}, std::move(exclusiveTimeline) }
    );
    return Submission{ timeline, 1, std::move(resource) };
}
Submission Subroutine::submit() const {
    return submit({});
}

void Subroutine::onDestroy() {
    if (subroutine) {
        subroutine.reset();
    }
}

Subroutine::Subroutine(Subroutine&&) noexcept = default;
Subroutine& Subroutine::operator=(Subroutine&&) = default;

Subroutine::Subroutine(
    ContextHandle context,
    SubroutineHandle subroutine,
    bool simultaneous_use)
    : Resource(std::move(context))
    , subroutine(std::move(subroutine))
    , simultaneous_use(simultaneous_use)
{}
Subroutine::~Subroutine() {
    onDestroy();
}

SubroutineBuilder::operator bool() const {
    return static_cast<bool>(subroutine);
}

SubroutineBuilder& SubroutineBuilder::addCommand(const Command& command) & {
    if (!*this)
        throw std::runtime_error("SubroutineBuilder has already finished!");

    command.record(subroutine->command);
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
    vulkan::checkResult(context->fnTable.vkEndCommandBuffer(subroutine->command.buffer));
    return Subroutine(std::move(context), std::move(subroutine), simultaneous_use);
}

SubroutineBuilder::SubroutineBuilder(SubroutineBuilder&&) noexcept = default;
SubroutineBuilder& SubroutineBuilder::operator=(SubroutineBuilder&&) noexcept = default;

SubroutineBuilder::SubroutineBuilder(ContextHandle context, bool simultaneous_use)
    : context(std::move(context))
    , subroutine(vulkan::beginSubroutine(this->context, simultaneous_use))
    , simultaneous_use(simultaneous_use)
{}
SubroutineBuilder::~SubroutineBuilder() = default;

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

void Timeline::onDestroy() {
    if (timeline) {
        auto& context = getContext();
        context->fnTable.vkDestroySemaphore(
            context->device, timeline->semaphore, nullptr);
        timeline.reset();
    }
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
    onDestroy();
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

template <class T>
const T* vectorBackOffset(const std::vector<T>& vec) {
    //we return the offset, when added to the base address of vector/array
    //will point to the currently last element in vec
    //NOTE: This is not a valid pointer and reading from it will likely crash!
    return reinterpret_cast<const T*>(static_cast<uint64_t>(vec.size()));
}

template<class T>
void applyVecOffset(const T* &offset, const std::vector<T>& vec) {
    offset = vec.data() + reinterpret_cast<uint64_t>(offset);
}

}

struct SequenceBuilder::pImp {
    //for recording commands
    VkCommandPool pool;
    vulkan::Command recordingCmd = {};
    std::vector<VkCommandBuffer> recordedBuffers = {};
    
    //submission infos
    std::vector<VkSubmitInfo2> submitInfos = {};
    std::vector<VkCommandBufferSubmitInfo> cmdBufSubmitInfos = {};
    std::vector<VkSemaphoreSubmitInfo> semaphoreWaits = {};
    std::vector<VkSemaphoreSubmitInfo> semaphoreSignals = {};

    //local sempahore for signaling
    uint64_t currentValue = 0; //value to wait for in the next batch
    std::unique_ptr<Timeline> exclusiveTimeline;
    Timeline& timeline;
    VkSemaphore semaphore;

    const vulkan::Context& context;

    void finishRecording() {
        //any buffer to finish
        if (!recordingCmd.buffer)
            return;

        //end recording
        vulkan::checkResult(context.fnTable.vkEndCommandBuffer(
            recordingCmd.buffer));
        cmdBufSubmitInfos.push_back({
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
            .commandBuffer = recordingCmd.buffer
        });
        recordedBuffers.push_back(recordingCmd.buffer);

        //reset recording command buffer
        recordingCmd = {};
    }

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
        _pImp->submitInfos.back().commandBufferInfoCount += 1;
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
    _pImp->cmdBufSubmitInfos.push_back({
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .commandBuffer = cmd.buffer
    });
    _pImp->submitInfos.back().commandBufferInfoCount += 1;

    return *this;
}

SequenceBuilder& SequenceBuilder::NextStep() & {
    if (!*this)
        throw std::runtime_error("SequenceBuilder has already finished!");

    //finish previous command buffer
    _pImp->finishRecording();

    //create new submission
    //we cannot yet store valid pointers. As the vectors grow the current ones
    //may become invalid. Their offset to the beginning, however, will stay
    //costant as we won't change their relative order. By storing this offset
    //we later only need to add the base address to get valid pointers back.
    //Just keep in mind that during construction reading the pointers will crash
    _pImp->submitInfos.push_back(VkSubmitInfo2{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .waitSemaphoreInfoCount = 1,
        .pWaitSemaphoreInfos = vectorBackOffset(_pImp->semaphoreWaits),
        .commandBufferInfoCount = 0,
        .pCommandBufferInfos = vectorBackOffset(_pImp->cmdBufSubmitInfos),
        .signalSemaphoreInfoCount = 1,
        .pSignalSemaphoreInfos = vectorBackOffset(_pImp->semaphoreSignals)
        });
    _pImp->semaphoreWaits.push_back(VkSemaphoreSubmitInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .semaphore = _pImp->semaphore,
        .value = _pImp->currentValue,
        .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT_KHR
    });
    _pImp->currentValue += 1;
    _pImp->semaphoreSignals.push_back(VkSemaphoreSubmitInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .semaphore = _pImp->semaphore,
        .value = _pImp->currentValue,
        .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT
    });

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
    if (_pImp->submitInfos.back().commandBufferInfoCount > 0)
        NextStep();

    //update wait/signal values
    //they are always the last ones
    _pImp->semaphoreWaits.back().value = value;
    _pImp->semaphoreSignals.back().value = value + 1;
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
    if (_pImp->submitInfos.back().commandBufferInfoCount > 0)
        NextStep();

    _pImp->submitInfos.back().waitSemaphoreInfoCount += 1;
    //ensure the last element in wait values/semaphores is the signaling timeline:
    // ... | back() | => ... | value | back() |
    // i.e. insert in second to last place
    //This is so WaitFor(uint64_t) wait works by simply alter the last value
    _pImp->semaphoreWaits.insert(
        _pImp->semaphoreWaits.end() - 1,
        VkSemaphoreSubmitInfo{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .semaphore = timeline.getTimeline().semaphore,
            .value = value,
            .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT_KHR
        }
    );

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
    //since we earlier stored the offsets, we only need to add the base address
    for (auto& submit : _pImp->submitInfos) {
        applyVecOffset(submit.pWaitSemaphoreInfos, _pImp->semaphoreWaits);
        applyVecOffset(submit.pCommandBufferInfos, _pImp->cmdBufSubmitInfos);
        applyVecOffset(submit.pSignalSemaphoreInfos, _pImp->semaphoreSignals);
    }

    //submit
    vulkan::checkResult(_pImp->context.fnTable.vkQueueSubmit2(
        _pImp->context.queue,
        static_cast<uint32_t>(_pImp->submitInfos.size()),
        _pImp->submitInfos.data(),
        nullptr
    ));

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
    for (auto& submit : _pImp->submitInfos) {
        //print wait values
        auto nWaits = submit.waitSemaphoreInfoCount;
        auto pWait = submit.pWaitSemaphoreInfos;
        applyVecOffset(pWait, _pImp->semaphoreWaits);
        for (auto i = 0; i < nWaits; ++i, ++pWait) {
            out << pWait->semaphore << '(' << pWait->value << ") ";
        }
        //print #commands/subroutines
        out << "-> " << submit.commandBufferInfoCount << ") -> ";
        //print signal values
        auto pSign = submit.pSignalSemaphoreInfos;
        applyVecOffset(pSign, _pImp->semaphoreSignals);
        out << pSign->semaphore << '(' << pSign->value << ")\n";
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
        vulkan::Command wrapper{ cmd };
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
        vulkan::Command wrapper{ cmd };
        emitter(wrapper);
    });
}

}
