#include "vk/instance.hpp"

#include <array>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "volk.h"

#include "hephaistos/debug.hpp"
#include "hephaistos/version.hpp"
#include "vk/result.hpp"

namespace {

//shared_ptr seems a bit heavy and since we only need the instance handle here
//we'll do the counting ourselves. Note that this is not thread safe, but I
//can't imagine why it'd be a good idea to create multiple contexts in parallel
uint32_t instanceReferenceCount = 0;
VkInstance instance;
VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;

//Debug state
PFN_vkDebugUtilsMessengerCallbackEXT debugCallback = nullptr;
std::vector<VkValidationFeatureEnableEXT> debugEnable = {};
std::vector< VkValidationFeatureDisableEXT> debugDisable = {};

constexpr auto DebugMessageSeverity =
    VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
    VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
    VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
    VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
constexpr auto DebugMessageType =
    VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
    VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
    VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT |
    VK_DEBUG_UTILS_MESSAGE_TYPE_DEVICE_ADDRESS_BINDING_BIT_EXT;

constexpr auto InstanceLayers = std::to_array({
    "VK_LAYER_KHRONOS_validation"
});
constexpr auto InstanceExtensions = std::to_array({
    VK_EXT_DEBUG_UTILS_EXTENSION_NAME
});

static VKAPI_ATTR VkBool32 VKAPI_CALL defaultDebugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData)
{

    //Create prefix
    std::string prefix("");
    if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT) prefix = "[VERB]";
    if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) prefix = "[INFO]";
    if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) prefix = "[WARN]";
    if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) prefix = "[ERR]";

    //Create message
    std::stringstream message;
    message << prefix << "(" << pCallbackData->messageIdNumber << ": " << pCallbackData->pMessageIdName << ") " << pCallbackData->pMessage;

    //Output
    if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        std::cerr << message.str() << std::endl;
    }
    else {
        std::cout << message.str() << std::endl;
    }
    fflush(stdout);

    //By returning false, the program will continue
    return VK_FALSE;
}

}

namespace hephaistos::vulkan {

void setInstanceDebugState(
    std::span<VkValidationFeatureEnableEXT> enable,
    std::span<VkValidationFeatureDisableEXT> disable,
    PFN_vkDebugUtilsMessengerCallbackEXT callback
) {
    debugEnable = std::vector<VkValidationFeatureEnableEXT>(
        enable.begin(), enable.end()
    );
    debugDisable = std::vector<VkValidationFeatureDisableEXT>(
        disable.begin(), disable.end()
    );
    debugCallback = callback;
}

VkInstance getInstance() {
	//singleton check
	if (instanceReferenceCount++ > 0)
		return instance;

	//Init volk if necessary
	if (volkGetInstanceVersion() == 0 && volkInitialize() != VK_SUCCESS)
		throw std::runtime_error("Vulkan is not supported in this system!");

    //check if debug available if requested
    bool debugEnabled = !debugEnable.empty() || !debugDisable.empty();
    if (debugEnabled && !isDebugAvailable())
        throw std::runtime_error("Debug was enabled but is not available!");

	auto version = VK_MAKE_VERSION(VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH);

    //configure instance
    VkApplicationInfo appInfo{
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "hephaistos",
        .applicationVersion = version,
        .pEngineName = "hephaistos",
        .engineVersion = version,
        .apiVersion = VK_API_VERSION_1_2
    };
    VkInstanceCreateInfo instInfo{
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &appInfo
    };
    VkValidationFeaturesEXT validation{
        .sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT,
        .enabledValidationFeatureCount = static_cast<uint32_t>(debugEnable.size()),
        .pEnabledValidationFeatures = debugEnable.data(),
        .disabledValidationFeatureCount = static_cast<uint32_t>(debugDisable.size()),
        .pDisabledValidationFeatures = debugDisable.data()
    };
    //optionally enable debug
    if (debugEnabled) {
        //add layer to instance creation info
        instInfo.enabledLayerCount = static_cast<uint32_t>(InstanceLayers.size());
        instInfo.ppEnabledLayerNames = InstanceLayers.data();
        instInfo.enabledExtensionCount = static_cast<uint32_t>(InstanceExtensions.size());
        instInfo.ppEnabledExtensionNames = InstanceExtensions.data();
        //enable validation features
        instInfo.pNext = &validation;
    }

    //create instance
    vulkan::checkResult(vkCreateInstance(&instInfo, nullptr, &instance));
    volkLoadInstanceOnly(instance);

    //optionally create debug messenger
    if (debugEnabled) {
        if (debugCallback == nullptr)
            debugCallback = defaultDebugCallback;

        VkDebugUtilsMessengerCreateInfoEXT debugInfo{
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            .messageSeverity = DebugMessageSeverity,
            .messageType = DebugMessageType,
            .pfnUserCallback = debugCallback
        };
        vulkan::checkResult(vkCreateDebugUtilsMessengerEXT(
            instance, &debugInfo, nullptr, &messenger));
    }

    //done
    return instance;
}

void returnInstance() {
    //destroy instance if recount reaches zero
    if (--instanceReferenceCount == 0) {
        vkDestroyDebugUtilsMessengerEXT(instance, messenger, nullptr);
        vkDestroyInstance(instance, nullptr);
    }
}

}
