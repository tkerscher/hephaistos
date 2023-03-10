#pragma once

#include "volk.h"
#include "hephaistos/config.hpp"

namespace hephaistos::vulkan {

struct Context {
    VkPhysicalDevice physicalDevice;
	VkDevice device;
	VkQueue queue;

	VkCommandPool cmdPool;
	VkPipelineCache cache;

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
