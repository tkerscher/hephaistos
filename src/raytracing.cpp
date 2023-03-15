#include "hephaistos/raytracing.hpp"

#include <algorithm>
#include <array>

#include "volk.h"

#include "vk/types.hpp"
#include "vk/result.hpp"
#include "vk/util.hpp"

namespace hephaistos {

/*********************************** EXTENSION ********************************/

namespace {

constexpr auto ExtensionName = "Raytracing";

//! MUST BE SORTED FOR std::includes !//
constexpr auto DeviceExtensions = std::to_array({
	VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
	VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
	VK_KHR_RAY_QUERY_EXTENSION_NAME
});

}

bool isRaytracingSupported(const DeviceHandle& device) {
	//We have to check two things:
	// - Extensions supported
	// - Features suppported

	//nullcheck
	if (!device)
		return false;

	//Check extension support
	if (!std::includes(
		device->supportedExtensions.begin(),
		device->supportedExtensions.end(),
		DeviceExtensions.begin(),
		DeviceExtensions.end()))
	{
		return false;
	}

	//Query features
	VkPhysicalDeviceBufferDeviceAddressFeatures addressFeatures{
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES
	};
	VkPhysicalDeviceRayQueryFeaturesKHR queryFeatures{
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR,
		.pNext = &addressFeatures
	};
	VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeatures{
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
		.pNext = &queryFeatures
	};
	VkPhysicalDeviceFeatures2 features{
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
		.pNext = &accelerationStructureFeatures
	};
	vkGetPhysicalDeviceFeatures2(device->device, &features);

	//check features
	if (!addressFeatures.bufferDeviceAddress)
		return false;
	if (!queryFeatures.rayQuery)
		return false;
	if (!accelerationStructureFeatures.accelerationStructure)
		return false;

	//Everything checked
	return true;
}
bool isRaytracingEnabled(const ContextHandle& context) {
	//to shorten things
	auto& ext = context->extensions;
	std::string_view name{ ExtensionName };
	return std::find(ext.begin(), ext.end(), name) != ext.end();
}

class RaytracingExtension : public Extension {
public:
	bool isDeviceSupported(const DeviceHandle& device) const override {
		return isRaytracingSupported(device);
	}
	std::string_view getExtensionName() const override {
		return ExtensionName;
	}
	std::span<const char* const> getDeviceExtensions() const override {
		return DeviceExtensions;
	}
	void* chain(void* pNext) override {
		addressFeatures.pNext = pNext;
		return static_cast<void*>(&accelerationStructureFeatures);
	}

	RaytracingExtension() = default;
	virtual ~RaytracingExtension() = default;

private:
	VkPhysicalDeviceBufferDeviceAddressFeatures addressFeatures{
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
		.bufferDeviceAddress = VK_TRUE
	};
	VkPhysicalDeviceRayQueryFeaturesKHR queryFeatures{
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR,
		.pNext = &addressFeatures,
		.rayQuery = VK_TRUE
	};
	VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeatures{
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
		.pNext = &queryFeatures,
		.accelerationStructure = VK_TRUE
	};
};
ExtensionHandle createRaytracingExtension() {
	return std::make_unique<RaytracingExtension>();
}

/*************************** ACCELERATION STRUCTURE ***************************/

//TODO

}
