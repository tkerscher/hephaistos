#include "vk/types.hpp"

#include "vk/result.hpp"

namespace hephaistos::vulkan {

BufferHandle createBuffer(
	const ContextHandle& context,
	uint64_t size,
	VkBufferUsageFlags usage,
	VmaAllocationCreateFlags flags)
{
	BufferHandle result{ new Buffer({0,0,{},*context}), destroyBuffer };

	VkBufferCreateInfo bufferInfo{
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage,
	};
	VmaAllocationCreateInfo allocInfo{
		.flags = flags,
		.usage = VMA_MEMORY_USAGE_AUTO
	};

	checkResult(vmaCreateBuffer(
		context->allocator,
		&bufferInfo,
		&allocInfo,
		&result->buffer,
		&result->allocation,
		&result->allocInfo));

	return result;
}
void destroyBuffer(Buffer* buffer) {
	if (!buffer)
		return;
	vmaDestroyBuffer(buffer->context.allocator, buffer->buffer, buffer->allocation);
}

CommandHandle createCommand(
	const ContextHandle& context,
	bool startRecording)
{
	CommandHandle result{
		new Command({nullptr, *context}),
		destroyCommand
	};

	VkCommandBufferAllocateInfo info{
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = context->cmdPool,
		.commandBufferCount = 1 
	};
	checkResult(context->fnTable.vkAllocateCommandBuffers(
		context->device, &info, &result->buffer));

	if (startRecording) {
		VkCommandBufferBeginInfo begin{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT
		};
		checkResult(context->fnTable.vkBeginCommandBuffer(result->buffer, &begin));
	}

	return result;
}
void destroyCommand(Command* command) {
	if (!command)
		return;

	command->context.fnTable.vkFreeCommandBuffers(
		command->context.device,
		command->context.cmdPool,
		1, &command->buffer);
}

}
