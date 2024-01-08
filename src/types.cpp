#include "hephaistos/types.hpp"

#include "vk/types.hpp"

namespace hephaistos {

namespace  {
	
auto ExtensionName = "Types";

constexpr uint32_t toBitFlags(const TypeSupport& types) {
	uint32_t v = 0;

	v += types.float64 << 0;
	v += types.float16 << 1;

	v += types.int64 << 2;
	v += types.int16 << 3;
	v += types.int8 << 4;

	return v;
}

class TypesExtension : public Extension {
public:
	bool isDeviceSupported(const DeviceHandle& device) const override {
		auto supported = toBitFlags(getSupportedTypes(device));
		return (supported & requiredFlags) == requiredFlags;
	}
	std::string_view getExtensionName() const override {
		return ExtensionName;
	}
	std::span<const char* const> getDeviceExtensions() const override {
		return {};
	}
	void* chain(void* pNext) override {
		//does not partake in chain -> skip
		return pNext;
	}

	TypesExtension(const TypeSupport& types)
		: requiredFlags(toBitFlags(types))
	{}
	virtual ~TypesExtension() = default;

private:
	uint32_t requiredFlags;
};

TypeSupport createTypeSupport(VkPhysicalDevice device) {
	VkPhysicalDeviceVulkan12Features features12{
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES
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
		!!features12.shaderInt8
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

}
