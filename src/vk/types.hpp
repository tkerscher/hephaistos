#pragma once

#include "volk.h"
#include "vk_mem_alloc.h"
#include "hephaistos/config.hpp"

namespace hephaistos::vulkan {

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

}
