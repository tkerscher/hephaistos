#include "hephaistos/context.hpp"

#include <array>
#include <stdexcept>

#include "hephaistos/handles.hpp"
#include "hephaistos/version.hpp"
#include "vk/result.hpp"
#include "vk/types.hpp"

#ifdef HEPHAISTOS_DEBUG
#include <iostream>
#include <sstream>

/************************************ DEBUG **********************************/

namespace {

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
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
constexpr auto EnabledValidationFeatures = std::to_array({
	VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT,
	VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT,
	VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT
	});

constexpr auto InstanceLayers = std::to_array({
	"VK_LAYER_KHRONOS_validation"
});
constexpr auto InstanceExtensions = std::to_array({
	VK_EXT_DEBUG_UTILS_EXTENSION_NAME
});

}
#else
namespace {

constexpr std::array<const char*, 0> InstanceLayers{};
constexpr std::array<const char*, 0> InstanceExtensions{};

}
#endif

namespace hephaistos {

/*********************************** INSTANCE ********************************/

namespace {

//shared_ptr seems a bit heavy and since we only need the instance handle here
//we'll do the counting ourselves. Note that this is not thread safe, but I
//can't imagine why it'd be a good idea to create multiple contexts in parallel
uint32_t instanceReferenceCount = 0;
VkInstance instance;
#ifdef HEPHAISTOS_DEBUG
VkDebugUtilsMessengerEXT messenger;
#endif

VkInstance getInstance() {
	//singleton check
	if (instanceReferenceCount++ > 0)
		return instance;

	//Init volk if necessary
	if (volkGetInstanceVersion() == 0 && volkInitialize() != VK_SUCCESS)
		throw std::runtime_error("Vulkan is not supported in this system!");

	auto version = VK_MAKE_VERSION(VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH);

	//configure instance
	VkApplicationInfo appInfo{
		.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO,
		.pApplicationName   = "hephaistos",
		.applicationVersion = version,
		.pEngineName        = "hephaistos",
		.engineVersion      = version,
		.apiVersion         = VK_API_VERSION_1_2
	};
	VkInstanceCreateInfo instInfo{
		.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
		.pApplicationInfo        = &appInfo,
		.enabledLayerCount       = static_cast<uint32_t>(InstanceLayers.size()),
		.ppEnabledLayerNames     = InstanceLayers.data(),
		.enabledExtensionCount   = static_cast<uint32_t>(InstanceExtensions.size()),
		.ppEnabledExtensionNames = InstanceExtensions.data()
	};
#ifdef HEPHAISTOS_DEBUG
	VkValidationFeaturesEXT validation{
		.sType                         = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT,
		.enabledValidationFeatureCount = static_cast<uint32_t>(EnabledValidationFeatures.size()),
		.pEnabledValidationFeatures    = EnabledValidationFeatures.data()
	};
	instInfo.pNext = &validation;
#endif

	//create instance
	vulkan::checkResult(vkCreateInstance(&instInfo, nullptr, &instance));
	volkLoadInstanceOnly(instance);

	//debug messenger
#ifdef HEPHAISTOS_DEBUG
	VkDebugUtilsMessengerCreateInfoEXT debugInfo{
		.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
		.messageSeverity = DebugMessageSeverity,
		.messageType = DebugMessageType,
		.pfnUserCallback = debugCallback
	};
	vulkan::checkResult(vkCreateDebugUtilsMessengerEXT(
		instance, &debugInfo, nullptr, &messenger));
#endif

	//done
	return instance;
}

void returnInstance() {
	//destroy instance if ref counter reaches 0
	if (--instanceReferenceCount == 0) {
#ifdef HEPHAISTOS_DEBUG
		vkDestroyDebugUtilsMessengerEXT(instance, messenger, nullptr);
#endif
		vkDestroyInstance(instance, nullptr);
	}
}

}

bool isVulkanAvailable() {
	//Check if volk is initialized or try to do so
	return (volkGetInstanceVersion() != 0) || (volkInitialize() == VK_SUCCESS);
}

/************************************ DEVICE *********************************/

namespace {

constexpr VkQueueFlags QueueFlags = VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
constexpr float QueuePriority = 1.0f;

bool isDeviceSuitable(VkPhysicalDevice device) {
	//Check for the following things:
	//	- Compute queue
	//	- timeline semaphore support

	//Check for timeline support
	{
		VkPhysicalDeviceTimelineSemaphoreFeatures timeline{
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES
		};
		VkPhysicalDeviceFeatures2 features{
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
			.pNext = &timeline
		};
		vkGetPhysicalDeviceFeatures2(device, &features);
		if (!timeline.timelineSemaphore)
			return false;
	}

	//check for compute queue
	{
		uint32_t count;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
		std::vector<VkQueueFamilyProperties> props(count);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &count, props.data());
		bool found = false;
		for (auto& queue : props) {
			if ((queue.queueFlags & QueueFlags) == QueueFlags) {
				found = true;
				break;
			}
		}
		if (!found)
			return false;
	}

	//Everything checked
	return true;
}

void destroyDevice(vulkan::Device* device) {
	delete device;
	returnInstance();
}

DeviceHandle createDevice(VkPhysicalDevice device) {
	//increment instance ref count
	getInstance();
	return DeviceHandle{ new vulkan::Device{ device }, destroyDevice };
};

DeviceInfo createInfo(VkPhysicalDevice device) {
	//retrieve props
	VkPhysicalDeviceProperties props;
	vkGetPhysicalDeviceProperties(device, &props);

	//return and build info
	return {
		.name       = props.deviceName,
		.isDiscrete = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
	};
}

}

std::vector<DeviceHandle> enumerateDevices() {
	//get instance
	auto instance = getInstance();

	//get all devices
	uint32_t count;
	vkEnumeratePhysicalDevices(instance, &count, nullptr);
	std::vector<VkPhysicalDevice> devices(count);
	vkEnumeratePhysicalDevices(instance, &count, devices.data());

    //transform supported devices
    std::vector<DeviceHandle> result;
    for (auto device : devices) {
        if (isDeviceSuitable(device))
            result.push_back(createDevice(device));
    }

	//we only needed the instance for the enumerate command
	//device handles have their own refs so return it
	returnInstance();

    //done
    return result;
}

DeviceHandle getDevice(const ContextHandle& context) {
	return createDevice(context->physicalDevice);
}

DeviceInfo getDeviceInfo(const DeviceHandle& device) {
	return createInfo(device->device);
}

DeviceInfo getDeviceInfo(const ContextHandle& context) {
	return createInfo(context->physicalDevice);
}

/*********************************** CONTEXT *********************************/

namespace {

void destroyContext(vulkan::Context* context) {
	context->fnTable.vkDestroyPipelineCache(context->device, context->cache, nullptr);
	context->fnTable.vkDestroyCommandPool(context->device, context->cmdPool, nullptr);
	context->fnTable.vkDestroyDevice(context->device, nullptr);
	returnInstance();
}

ContextHandle createContext(VkPhysicalDevice device) {
	//Allocate memory
	ContextHandle context{ new vulkan::Context, destroyContext };
	context->physicalDevice = device;

	//query queue family
	uint32_t family = 0;
	{
		uint32_t count;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
		std::vector<VkQueueFamilyProperties> props(count);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &count, props.data());
		for (; family < count; ++family) {
			if (props[family].queueFlags & QueueFlags)
				break;
		}
	}

	//Create logical device
	{
		VkDeviceQueueCreateInfo queueInfo{
			.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
			.queueFamilyIndex = family,
			.queueCount = 1,
			.pQueuePriorities = &QueuePriority
		};
		VkPhysicalDeviceTimelineSemaphoreFeatures timeline{
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
			.timelineSemaphore = VK_TRUE
		};
		VkDeviceCreateInfo deviceInfo{
			.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
			.pNext = &timeline,
			.queueCreateInfoCount = 1,
			.pQueueCreateInfos = &queueInfo,
		};
		vulkan::checkResult(vkCreateDevice(device, &deviceInfo, nullptr, &context->device));
	}

	//load device functions
	volkLoadDeviceTable(&context->fnTable, context->device);
	//get queue
	context->fnTable.vkGetDeviceQueue(context->device, family, 0, &context->queue);

	//create command pool
	{
		VkCommandPoolCreateInfo poolInfo{
			.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			.queueFamilyIndex = family
		};
		vulkan::checkResult(context->fnTable.vkCreateCommandPool(
			context->device, &poolInfo, nullptr, &context->cmdPool));
	}

	//Create pipeline cache
	{
		VkPipelineCacheCreateInfo cacheInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO
		};
		vulkan::checkResult(context->fnTable.vkCreatePipelineCache(
			context->device, &cacheInfo, nullptr, &context->cache));
	}

	//Done
	return context;
}

}

ContextHandle createContext() {
	//new handle -> refs to instance
	auto instance = getInstance();

	//enumerate devices
	uint32_t count;
	vkEnumeratePhysicalDevices(instance, &count, nullptr);
	std::vector<VkPhysicalDevice> devices(count);
	vkEnumeratePhysicalDevices(instance, &count, devices.data());

	//select first suitable discrete GPU
	VkPhysicalDevice fallback = nullptr;
	VkPhysicalDeviceProperties props;
	for (auto device : devices) {
		if (isDeviceSuitable(device)) {
			//Remember last suitable device if no discrete present
			fallback = device;
			vkGetPhysicalDeviceProperties(device, &props);
			if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
				return createContext(device);
		}
	}

	//no discrete gpu
	if (fallback)
		return createContext(fallback);
	else
		throw std::runtime_error("No suitable device available!");
}

ContextHandle createContext(const DeviceHandle& device) {
	if (!device)
		throw std::runtime_error("Device is empty!");

	//new ref
	getInstance();
	return createContext(device->device);
}

}
