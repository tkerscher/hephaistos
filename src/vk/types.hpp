#pragma once

#include "volk.h"
#include "vk_mem_alloc.h"
#include "hephaistos/config.hpp"
#include "hephaistos/handles.hpp"

namespace hephaistos::vulkan {

struct Buffer {
	VkBuffer buffer;
	VmaAllocation allocation;
	VmaAllocationInfo allocInfo;

	const Context& context;
};

struct Context {
    VkPhysicalDevice physicalDevice;
	VkDevice device;
	VkQueue queue;

	VkCommandPool cmdPool;
	VkPipelineCache cache;

	VmaAllocator allocator;

	VolkDeviceTable fnTable;
};

struct Device {
    VkPhysicalDevice device;
};

struct Instance {
    VkInstance instance;
#ifdef HEPHAISTOS_DEBUG
    VkDebugUtilsMessengerEXT messenger;
#endif
};

[[nodiscard]] BufferHandle createBuffer(
	const ContextHandle& handle,
	uint64_t size,
	VkBufferUsageFlags usage,
	VmaAllocationCreateFlags flags);
void destroyBuffer(Buffer* buffer);

}
