#pragma once

#include <span>

#include "volk.h"

namespace hephaistos::vulkan {

//Debug -> save params for later lazy creation
void setInstanceDebugState(
	std::span<VkValidationFeatureEnableEXT> enable,
	std::span<VkValidationFeatureDisableEXT> disable,
	PFN_vkDebugUtilsMessengerCallbackEXT callback
);

VkInstance getInstance();
void returnInstance();

}
