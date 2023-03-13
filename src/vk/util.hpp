#pragma once

#include "vk/result.hpp"
#include "vk/types.hpp"

namespace hephaistos::vulkan {

template<class Func>
void oneTimeSubmit(Context& context, const Func& func) {
	//Allocate command buffer and begin recording
	VkCommandBuffer cmdBuffer;
	VkCommandBufferAllocateInfo allocInfo{
		.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool        = context.oneTimeSubmitPool,
		.commandBufferCount = 1
	};
	checkResult(context.fnTable.vkAllocateCommandBuffers(
		context.device, &allocInfo, &cmdBuffer));

	//Start recording
	VkCommandBufferBeginInfo beginInfo{
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
	};
	checkResult(context.fnTable.vkBeginCommandBuffer(
		cmdBuffer, &beginInfo));

	//record stuff
	func(cmdBuffer);

	//End recording
	checkResult(context.fnTable.vkEndCommandBuffer(cmdBuffer));

	//Submit with fence so we can wait for it to finish
	VkSubmitInfo submitInfo{
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.commandBufferCount = 1,
		.pCommandBuffers = &cmdBuffer
	};
	checkResult(context.fnTable.vkQueueSubmit(
		context.queue, 1, &submitInfo, context.oneTimeSubmitFence));

	//wait for it to finish
	checkResult(context.fnTable.vkWaitForFences(
		context.device, 1, &context.oneTimeSubmitFence, VK_TRUE, UINT64_MAX));
	
	//reset fence for next use
	checkResult(context.fnTable.vkResetFences(
		context.device, 1, &context.oneTimeSubmitFence));
}

}
