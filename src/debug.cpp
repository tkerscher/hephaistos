#include "hephaistos/debug.hpp"

#include <array>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "volk.h"

#include "vk/instance.hpp"
#include "vk/result.hpp"
#include "vk/types.hpp"

#include "hephaistos/context.hpp"

namespace hephaistos {

/********************************** CALLBACK **********************************/

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

/******************************** DEVICE FAULT ********************************/

namespace {

constexpr auto DeviceFaultExtensionName = "DeviceFault";

constexpr auto DeviceFaultExtensions = std::to_array({
	VK_EXT_DEVICE_FAULT_EXTENSION_NAME
});

class DeviceFaultExtension final : public Extension {
public:
	bool isDeviceSupported(const DeviceHandle& device) const override {
		return isDeviceFaultExtensionSupported(device);
	}	
	std::string_view getExtensionName() const override {
		return DeviceFaultExtensionName;
	}
	std::span<const char* const> getDeviceExtensions() const override {
		return DeviceFaultExtensions;
	}
	void* chain(void* pNext) override {
		features.pNext = pNext;
		return &features;
	}
	void finalize(const ContextHandle& context) override {}

	DeviceFaultExtension() {
		features = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT,
			.deviceFault = VK_TRUE
		};
	}
	~DeviceFaultExtension() override = default;

private:
	VkPhysicalDeviceFaultFeaturesEXT features;
};
	
}

bool isDeviceFaultExtensionSupported(const DeviceHandle& device) {
	VkPhysicalDeviceFaultFeaturesEXT fault{
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT
	};
	VkPhysicalDeviceFeatures2 features{
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
		.pNext = &fault
	};
	vkGetPhysicalDeviceFeatures2(device->device, &features);

	return fault.deviceFault;
}

ExtensionHandle createDeviceFaultInfoExtension() {
	return std::make_unique<DeviceFaultExtension>();
}

DeviceFaultInfo getDeviceFaultInfo(const ContextHandle& context) {
	VkDeviceFaultCountsEXT counts{
		.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT
	};
	vulkan::checkResult(context->fnTable.vkGetDeviceFaultInfoEXT(
		context->device, &counts, nullptr));

	std::vector<VkDeviceFaultAddressInfoEXT> address{ counts.addressInfoCount };
	std::vector<VkDeviceFaultVendorInfoEXT> vendor{ counts.vendorInfoCount };
	VkDeviceFaultInfoEXT info{
		.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT,
		.pAddressInfos = address.data(),
		.pVendorInfos = vendor.data()
	};
	vulkan::checkResult(context->fnTable.vkGetDeviceFaultInfoEXT(
		context->device, &counts, &info));

	DeviceFaultInfo result{};
	result.description = info.description;
	for (auto& a : address) {
		result.addressInfo.emplace_back(
			static_cast<DeviceFaultAddressType>(static_cast<int32_t>(a.addressType)),
			a.reportedAddress,
			a.addressPrecision
		);
	}
	for (auto& v : vendor) {
		result.vendorInfo.emplace_back(
			v.description,
			v.vendorFaultCode,
			v.vendorFaultData
		);
	}

	return result;
}

}
