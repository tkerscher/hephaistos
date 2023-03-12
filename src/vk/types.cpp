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

}
