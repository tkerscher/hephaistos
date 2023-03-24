#include "hephaistos/command.hpp"

#include <vector>

#include "volk.h"

#include "vk/result.hpp"
#include "vk/types.hpp"

namespace hephaistos {

Command::~Command() = default;

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
	if (resources) {
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
		, pool(fetchCommandPool(*timeline.getContext()))
		, exclusiveTimeline(nullptr)
		, semaphore(timeline.getTimeline().semaphore)
		, timeline(timeline)
		, currentValue(value + 1)
		, context(*timeline.getContext())
	{
		commands.push_back({});
		waitValues.push_back(value);
		signalValues.push_back(value + 1);
	}
	pImp(ContextHandle context)
		: commands({})
		, pool(fetchCommandPool(*context))
		, exclusiveTimeline(std::make_unique<Timeline>(std::move(context)))
		, semaphore(exclusiveTimeline->getTimeline().semaphore)
		, timeline(*exclusiveTimeline)
		, currentValue(1)
		, context(*timeline.getContext())
	{
		commands.push_back({});
		waitValues.push_back(0);
		signalValues.push_back(1);
	}
};

SequenceBuilder& SequenceBuilder::And(const Command& command) {
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

SequenceBuilder& SequenceBuilder::Then(const Command& command) {
	//Finish previous command buffer
	auto& prevCmd = _pImp->commands.back();
	if (prevCmd.buffer)
		vulkan::checkResult(_pImp->context.fnTable.vkEndCommandBuffer(prevCmd.buffer));

	//add new level
	_pImp->commands.emplace_back();
	_pImp->waitValues.push_back(_pImp->currentValue);
	_pImp->currentValue++;
	_pImp->signalValues.push_back(_pImp->currentValue);

	return And(command);
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

	//build infos
	auto size = _pImp->commands.size();
	std::vector<VkTimelineSemaphoreSubmitInfo> timelineInfos(size);
	std::vector<VkSubmitInfo> submits(size);
	auto wPtr = _pImp->waitValues.data();
	auto sPtr = _pImp->signalValues.data();
	auto cPtr = _pImp->commands.data();
	auto tPtr = timelineInfos.data();
	for (auto i = 0u; i < submits.size(); ++i, ++wPtr, ++sPtr, ++cPtr, ++tPtr) {
		*tPtr = VkTimelineSemaphoreSubmitInfo{
			.sType                     = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
			.waitSemaphoreValueCount   = 1,
			.pWaitSemaphoreValues      = wPtr,
			.signalSemaphoreValueCount = 1,
			.pSignalSemaphoreValues    = sPtr
		};
		auto empty = cPtr->buffer == nullptr;
		submits[i] = VkSubmitInfo{
			.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.pNext                = tPtr,
			.waitSemaphoreCount   = 1,
			.pWaitSemaphores      = &_pImp->semaphore,
			.pWaitDstStageMask    = empty ? &EmptyStage : &cPtr->stage,
			.commandBufferCount   = empty ? 0u : 1u,
			.pCommandBuffers      = empty ? nullptr : &cPtr->buffer,
			.signalSemaphoreCount = 1,
			.pSignalSemaphores    = &_pImp->semaphore
		};
	}

	//submit
	vulkan::checkResult(_pImp->context.fnTable.vkQueueSubmit(
		_pImp->context.queue, submits.size(), submits.data(), nullptr));

	//collect data
	std::vector<VkCommandBuffer> cmdBuffers(_pImp->commands.size());
	for (int i = 0u; i < cmdBuffers.size(); ++i)
		cmdBuffers[i] = _pImp->commands[i].buffer;
	auto resource = std::make_unique<SubmissionResources>(
		_pImp->pool,
		std::move(cmdBuffers),
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

}
