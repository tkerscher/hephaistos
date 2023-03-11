#include "hephaistos/command.hpp"

#include <vector>

#include "volk.h"

#include "vk/result.hpp"
#include "vk/types.hpp"

namespace hephaistos {

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

/********************************* SEQUENCE **********************************/

struct SequenceBuilder::pImp {
	std::vector<std::vector<VkCommandBuffer>> commands;
	std::vector<VkPipelineStageFlags> stages;
	std::vector<uint64_t> waitValues;
	std::vector<uint64_t> signalValues;
	//the value to wait for in the next batch. Might differ from signalValues.back()
	uint64_t currentValue;

	VkSemaphore timeline;

	const vulkan::Context& context;

	pImp(const vulkan::Context& context, VkSemaphore timeline, uint64_t value)
		: commands({})
		, stages({ VK_PIPELINE_STAGE_NONE })
		, timeline(timeline)
		, currentValue(value + 1)
		, context(context)
	{
		commands.push_back({});
		waitValues.push_back(value);
		signalValues.push_back(value + 1);
	}
};

SequenceBuilder& SequenceBuilder::And(const CommandHandle& command) {
	if (&command->context != &_pImp->context)
		throw std::logic_error("Timeline and command must originate from the same context!");

	_pImp->commands.back().push_back(command->buffer);
	_pImp->stages.back() |= command->stage;
	return *this;
}

SequenceBuilder& SequenceBuilder::Then(const CommandHandle& command) {
	//add new level
	_pImp->commands.emplace_back();
	_pImp->stages.emplace_back(VK_PIPELINE_STAGE_NONE);
	_pImp->waitValues.push_back(_pImp->currentValue);
	_pImp->currentValue++;
	_pImp->signalValues.push_back(_pImp->currentValue);

	return And(command);
}

SequenceBuilder& SequenceBuilder::WaitFor(uint64_t value) {
	if (value < _pImp->currentValue)
		throw std::logic_error("Wait value too low! Timeline can only go forward!");

	//add new level
	_pImp->commands.emplace_back();
	_pImp->stages.emplace_back(VK_PIPELINE_STAGE_NONE);
	_pImp->waitValues.push_back(value);
	_pImp->currentValue = value + 1;
	_pImp->signalValues.push_back(value + 1);

	return *this;
}

uint64_t SequenceBuilder::Submit() {
	if (!_pImp)
		throw std::logic_error("A sequence can only be submitted once!");

	//build infos
	auto size = _pImp->commands.size();
	std::vector<VkTimelineSemaphoreSubmitInfo> timelineInfos(size);
	std::vector<VkSubmitInfo> submits(size);
	auto wPtr = _pImp->waitValues.data();
	auto sPtr = _pImp->signalValues.data();
	auto fPtr = _pImp->stages.data();
	auto tPtr = timelineInfos.data();
	for (auto i = 0u; i < submits.size(); ++i, ++wPtr, ++sPtr, ++fPtr, ++tPtr) {
		*tPtr = VkTimelineSemaphoreSubmitInfo{
			.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
			.waitSemaphoreValueCount = 1,
			.pWaitSemaphoreValues = wPtr,
			.signalSemaphoreValueCount = 1,
			.pSignalSemaphoreValues = sPtr
		};
		submits[i] = VkSubmitInfo{
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.pNext = tPtr,
			.waitSemaphoreCount = 1,
			.pWaitSemaphores = &_pImp->timeline,
			.pWaitDstStageMask = fPtr,
			.commandBufferCount = static_cast<uint32_t>(_pImp->commands[i].size()),
			.pCommandBuffers = _pImp->commands[i].data(),
			.signalSemaphoreCount = 1,
			.pSignalSemaphores = &_pImp->timeline
		};
	}

	//submit
	vulkan::checkResult(_pImp->context.fnTable.vkQueueSubmit(
		_pImp->context.queue, submits.size(), submits.data(), nullptr));

	//return final signal value
	return _pImp->currentValue;
}

SequenceBuilder::SequenceBuilder(Timeline& timeline, uint64_t startValue)
	: _pImp(new pImp(
		*timeline.getContext(),
		timeline.getTimeline().semaphore,
		startValue))
{}
SequenceBuilder::~SequenceBuilder() = default;

}
