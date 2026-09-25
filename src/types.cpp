#include "hephaistos/types.hpp"

#include <array>

#include "vk/types.hpp"

namespace hephaistos {

/*********************************** TYPES ***********************************/

namespace  {
	
constexpr auto TypesExtensionName = "Types";

constexpr uint32_t toBitFlags(const TypeSupport& types) {
	uint32_t v = 0;

	v += types.float64 << 0;
	v += types.float16 << 1;

	v += types.int64 << 2;
	v += types.int16 << 3;
	v += types.int8 << 4;

	v += types.buffer16BitAccess << 5;
	v += types.uniform16BitAccess << 6;
	v += types.pushConstant16BitAccess << 7;

	v += types.buffer8BitAccess << 8;
	v += types.uniform8BitAccess << 9;
	v += types.pushConstant8BitAccess << 10;

	return v;
}

class TypesExtension : public Extension {
public:
	bool isDeviceSupported(const DeviceHandle& device) const override {
		auto supported = toBitFlags(getSupportedTypes(device));
		return (supported & requiredFlags) == requiredFlags;
	}
	std::string_view getExtensionName() const override {
		return TypesExtensionName;
	}
	std::span<const char* const> getDeviceExtensions() const override {
		return {};
	}
	void* chain(void* pNext) override {
		//does not partake in chain -> skip
		return pNext;
	}
	void finalize(const ContextHandle& context) {}

	TypesExtension(const TypeSupport& types)
		: requiredFlags(toBitFlags(types))
	{}
	virtual ~TypesExtension() = default;

private:
	uint32_t requiredFlags;
};

TypeSupport createTypeSupport(VkPhysicalDevice device) {
	VkPhysicalDevice8BitStorageFeatures storage8{
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES
	};
	VkPhysicalDevice16BitStorageFeatures storage16{
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
		.pNext = &storage8
	};
	VkPhysicalDeviceVulkan12Features features12{
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
		.pNext = &storage16
	};
	VkPhysicalDeviceFeatures2 features2{
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
		.pNext = &features12
	};
	vkGetPhysicalDeviceFeatures2(device, &features2);

	return {
		!!features2.features.shaderFloat64,
		!!features12.shaderFloat16,
		!!features2.features.shaderInt64,
		!!features2.features.shaderInt16,
		!!features12.shaderInt8,
		!!storage16.storageBuffer16BitAccess,
		!!storage16.uniformAndStorageBuffer16BitAccess,
		!!storage16.storagePushConstant16,
		!!storage8.storageBuffer8BitAccess,
		!!storage8.uniformAndStorageBuffer8BitAccess,
		!!storage8.storagePushConstant8
	};
}

}

TypeSupport getSupportedTypes(const DeviceHandle& device) {
	//null check
	if (!device)
		return {};
	
	return createTypeSupport(device->device);
}

TypeSupport getSupportedTypes(const ContextHandle& context) {
	//null check
	if (!context)
		return {};

	return createTypeSupport(context->physicalDevice);
}

ExtensionHandle createTypeExtension(const TypeSupport& types) {
	return std::make_unique<TypesExtension>(types);
}

/************************************ FMA ************************************/

namespace {

constexpr auto FmaExtensionName = "FMA";

class FmaExtension : public Extension {
public:
	bool isDeviceSupported(const DeviceHandle& device) const override {
		auto supported = getFmaSupport(device);
		return
			(!required.float16 || supported.float16) &&
			(!required.float32 || supported.float32) &&
			(!required.float64 || supported.float64);
	}
	std::string_view getExtensionName() const override {
		return FmaExtensionName;
	}
	std::span<const char* const> getDeviceExtensions() const override {
		//this extension is automatically added during context creation
		//-> do not add it twice
		return {};
	}
	void* chain(void* pNext) override {
		//does not partake in chain -> skip
		return pNext;
	}
	void finalize(const ContextHandle& context) {}

	FmaExtension(const FmaSupport& fma) : required(fma) {}
	virtual ~FmaExtension() = default;

private:
	FmaSupport required;
};

FmaSupport queryFmaSupport(VkPhysicalDevice device) {
	VkPhysicalDeviceShaderFmaFeaturesKHR fma{
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FMA_FEATURES_KHR
	};
	VkPhysicalDeviceFeatures2 features{
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
		.pNext = &fma
	};
	vkGetPhysicalDeviceFeatures2(device, &features);

	return {
		!!fma.shaderFmaFloat16,
		!!fma.shaderFmaFloat32,
		!!fma.shaderFmaFloat64
	};
}

}

FmaSupport getFmaSupport(const DeviceHandle& device) {
	if (!device)
		return {};
	return queryFmaSupport(device->device);
}

FmaSupport getFmaSupport(const ContextHandle& context) {
	if (!context)
		return {};
	return queryFmaSupport(context->physicalDevice);
}

ExtensionHandle createFmaExtension(const FmaSupport& fma) {
	return std::make_unique<FmaExtension>(fma);
}

}
