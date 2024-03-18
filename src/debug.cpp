#include "hephaistos/debug.hpp"

#include <cstring>
#include <stdexcept>
#include <vector>

#include "volk.h"

#include "vk/instance.hpp"

namespace hephaistos {

namespace {

//user provided callback
DebugCallback debugCallback;

//converter callback
static VKAPI_ATTR VkBool32 VKAPI_CALL transformCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData)
{
	//convert vulkan format to out format
	auto sev = static_cast<DebugMessageSeverityFlagBits>(messageSeverity);
	DebugMessage message{
		.severity = sev,
		.pIdName = pCallbackData->pMessageIdName,
		.idNumber = pCallbackData->messageIdNumber,
		.pMessage = pCallbackData->pMessage
	};
	//call callback
	if (debugCallback) debugCallback(message);

	//By returning false, the program will continue
	return VK_FALSE;
}

}

bool isDebugAvailable() {
	//Init volk if necessary
	if (volkGetInstanceVersion() == 0 && volkInitialize() != VK_SUCCESS)
		throw std::runtime_error("Vulkan is not supported in this system!");

	//fetch all available layers
	uint32_t count;
	vkEnumerateInstanceLayerProperties(&count, nullptr);
	std::vector<VkLayerProperties> layers(count);
	vkEnumerateInstanceLayerProperties(&count, layers.data());
	//check if validation layer are available
	for (const auto& layer : layers) {
		if (strncmp(layer.layerName, "VK_LAYER_KHRONOS_validation", 28) == 0)
			return true;
	}
	//not found
	return false;
}

void configureDebug(DebugOptions options, DebugCallback callback) {
	//build features list
	std::vector< VkValidationFeatureEnableEXT> enable;
	std::vector< VkValidationFeatureDisableEXT> disable;
	if (options.enablePrint)
		enable.push_back(VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT);
	if (options.enableGPUValidation)
		enable.push_back(VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT);
	if (options.enableSynchronizationValidation)
		enable.push_back(VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT);
	if (!options.enableThreadSafetyValidation)
		disable.push_back(VK_VALIDATION_FEATURE_DISABLE_THREAD_SAFETY_EXT);
	if (options.enableAPIValidation) {
		enable.push_back(VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT);
	}
	else {
		disable.push_back(VK_VALIDATION_FEATURE_DISABLE_API_PARAMETERS_EXT);
		disable.push_back(VK_VALIDATION_FEATURE_DISABLE_OBJECT_LIFETIMES_EXT);
		if (!options.enableGPUValidation)
			disable.push_back(VK_VALIDATION_FEATURE_DISABLE_CORE_CHECKS_EXT);
		disable.push_back(VK_VALIDATION_FEATURE_DISABLE_UNIQUE_HANDLES_EXT);
	}
	//save callback
	debugCallback = callback;

	//set state
	PFN_vkDebugUtilsMessengerCallbackEXT pCallback;
	pCallback = debugCallback ? transformCallback : nullptr;
	vulkan::setInstanceDebugState(enable, disable, pCallback);
}

}
