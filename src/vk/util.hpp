#pragma once

#include "vk/result.hpp"
#include "vk/types.hpp"

namespace hephaistos::vulkan {

template<class Func>
void oneTimeSubmit(Context& context, const Func& func) {
    //Start recording
    VkCommandBufferBeginInfo beginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };
    checkResult(context.fnTable.vkBeginCommandBuffer(
        context.oneTimeSubmitBuffer, &beginInfo));

    //record stuff
    func(context.oneTimeSubmitBuffer);

    //End recording
    checkResult(context.fnTable.vkEndCommandBuffer(context.oneTimeSubmitBuffer));

    //Submit with fence so we can wait for it to finish
    VkSubmitInfo submitInfo{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &context.oneTimeSubmitBuffer
    };
    checkResult(context.fnTable.vkQueueSubmit(
        context.queue, 1, &submitInfo, context.oneTimeSubmitFence));

    //wait for it to finish
    checkResult(context.fnTable.vkWaitForFences(
        context.device, 1, &context.oneTimeSubmitFence, VK_TRUE, UINT64_MAX));
    
    //reset fence for next use
    checkResult(context.fnTable.vkResetFences(
        context.device, 1, &context.oneTimeSubmitFence));

    //reset command buffer via pool for next use
    checkResult(context.fnTable.vkResetCommandPool(
        context.device, context.oneTimeSubmitPool, 0));
}

}
