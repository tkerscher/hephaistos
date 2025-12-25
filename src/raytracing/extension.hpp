#pragma once

#include <array>
#include <span>

#include "volk.h"

#include "hephaistos/context.hpp"
#include "hephaistos/raytracing/extension.hpp"

namespace hephaistos {

class RayTracingExtension : public Extension {
public:
	bool isDeviceSupported(const DeviceHandle& device) const override;
	std::string_view getExtensionName() const override;
	std::span<const char* const> getDeviceExtensions() const override;
	void* chain(void* pNext) override;
	void finalize(const ContextHandle& context) override;

	RayTracingExtension(RayTracingFeatures features);
	virtual ~RayTracingExtension();

public:
	//cached properties
	RayTracingFeatures enabled;
	RayTracingProperties props;

	struct ShaderGroupProperties {
		uint32_t shaderGroupHandleSize;
		uint32_t shaderGroupBaseAlignment;
		uint32_t shaderGroupHandleAlignment;
	} shaderGroupProperties;

	uint32_t minAccelerationStructureScratchOffsetAlignment;

private:
	std::array<const char*, 6> extensions;
	uint32_t nExtension;
	//std::vector<const char*> enabledExtensionNames;

	VkPhysicalDeviceRayTracingInvocationReorderFeaturesEXT reorderFeatures;
	VkPhysicalDeviceRayTracingPositionFetchFeaturesKHR posFetchFeatures;
	VkPhysicalDeviceRayTracingPipelineFeaturesKHR pipelineFeatures;
	VkPhysicalDeviceRayQueryFeaturesKHR queryFeatures;
	VkPhysicalDeviceAccelerationStructureFeaturesKHR accStructFeatures;
};

}
