#include "hephaistos/raytracing/extension.hpp"
#include "raytracing/extension.hpp"

#include <array>

#include "volk.h"

#include "vk/types.hpp"

namespace hephaistos {

namespace {

constexpr auto RayTracing_ExtensionName = "RayTracing";

}

RayTracingFeatures getRayTracingFeatures(const DeviceHandle& device) {
	//null check
	if (!device) return {};

	//query features
    VkPhysicalDeviceRayTracingInvocationReorderFeaturesEXT reorderFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_EXT
    };
    VkPhysicalDeviceRayTracingPositionFetchFeaturesKHR posFetchFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_POSITION_FETCH_FEATURES_KHR,
        .pNext = &reorderFeatures
    };
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR pipelineFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
        .pNext = &posFetchFeatures
    };
    VkPhysicalDeviceRayQueryFeaturesKHR queryFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR,
        .pNext = &pipelineFeatures
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

    //check common required features
    if (!accelerationStructureFeatures.accelerationStructure)
        return {};

    //check optional features
    return {
        !!queryFeatures.rayQuery,
        !!pipelineFeatures.rayTracingPipeline,
        !!pipelineFeatures.rayTracingPipelineTraceRaysIndirect,
        !!posFetchFeatures.rayTracingPositionFetch,
        !!reorderFeatures.rayTracingInvocationReorder
    };
}

bool isRayTracingSupported(RayTracingFeatures features) {
    auto devices = enumerateDevices();
    return std::any_of(
        devices.begin(),
        devices.end(),
        [&features](const DeviceHandle& device) -> bool {
            return isRayTracingSupported(device, features);
        });
}

bool isRayTracingSupported(const DeviceHandle& device, RayTracingFeatures features) {
    auto supports = getRayTracingFeatures(device);

    //check if ray tracing is supported at all
    if (!supports.pipeline || !supports.query)
        return false;

    //check if all requested features are supported
    return (!features.query || supports.query) &&
        (!features.pipeline || supports.pipeline) &&
        (!features.indirectDispatch || supports.indirectDispatch) &&
        (!features.positionFetch || supports.positionFetch) &&
        (!features.hitObjects || supports.hitObjects);
}

RayTracingProperties getRayTracingProperties(const DeviceHandle& device) {
    //is ray tracing even supported?
    auto supports = getRayTracingFeatures(device);

    //query properties
    VkPhysicalDeviceRayTracingInvocationReorderPropertiesEXT reorderProps{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_PROPERTIES_EXT
    };
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR pipelineProps{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR,
        .pNext = &reorderProps
    };
    VkPhysicalDeviceAccelerationStructurePropertiesKHR accProps{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR,
        .pNext = &pipelineProps
    };
    VkPhysicalDeviceProperties2 props{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &accProps
    };
    vkGetPhysicalDeviceProperties2(device->device, &props);

    //assemble properties
    //values in the structs are undefined if the extension is not supported
    RayTracingProperties result{};
    if (supports.query || supports.pipeline) {
        result.maxGeometryCount = accProps.maxGeometryCount;
        result.maxInstanceCount = accProps.maxInstanceCount;
        result.maxPrimitiveCount = accProps.maxPrimitiveCount;
        result.maxAccelerationStructures = std::min(
            accProps.maxDescriptorSetAccelerationStructures,
            accProps.maxPerStageDescriptorAccelerationStructures
        );
    }
    if (supports.pipeline) {
        result.maxRayRecursionDepth = pipelineProps.maxRayRecursionDepth;
        result.maxRayDispatchCount = pipelineProps.maxRayDispatchInvocationCount;
        //result.maxRayHitAttributeSize = pipelineProps.maxRayHitAttributeSize;
        auto handleSize = pipelineProps.shaderGroupHandleSize;
        auto groupStride = pipelineProps.maxShaderGroupStride;
        result.maxShaderRecordSize = groupStride - handleSize;
    }
    if (supports.hitObjects) {
        result.canReorder =
            VK_RAY_TRACING_INVOCATION_REORDER_MODE_REORDER_EXT ==
            reorderProps.rayTracingInvocationReorderReorderingHint;
    }
    return result;
}

RayTracingProperties getCurrentRayTracingProperties(const ContextHandle& context) {
    auto ext = getExtension<RayTracingExtension>(*context, "RayTracing");
    if (!ext)
        return {};
    else
        return ext->props;
}

ExtensionHandle createRayTracingExtension(RayTracingFeatures enabled) {
    return std::make_unique<RayTracingExtension>(enabled);
}

RayTracingFeatures getEnabledRayTracing(const ContextHandle& context) {
    auto pExt = std::find_if(
        context->extensions.begin(),
        context->extensions.end(),
        [](const ExtensionHandle& h) -> bool {
            return h->getExtensionName() == RayTracing_ExtensionName;
        }
    );
    if (pExt == context->extensions.end())
        return {};
    else
        return dynamic_cast<RayTracingExtension*>(pExt->get())->enabled;
}

/********************************* EXTENSION **********************************/

bool RayTracingExtension::isDeviceSupported(const DeviceHandle& device) const {
    return isRayTracingSupported(device, enabled);
}

std::string_view RayTracingExtension::getExtensionName() const {
    return RayTracing_ExtensionName;
}

std::span<const char* const> RayTracingExtension::getDeviceExtensions() const {
    return { extensions.begin(), nExtension };
}

void* RayTracingExtension::chain(void* pNext) {
    accStructFeatures = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
        .pNext = pNext,
        .accelerationStructure = VK_TRUE
    };
    pNext = &accStructFeatures;

    if (enabled.query) {
        queryFeatures = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR,
            .pNext = pNext,
            .rayQuery = VK_TRUE
        };
        pNext = &queryFeatures;
    }
    if (enabled.pipeline) {
        pipelineFeatures = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
            .pNext = pNext,
            .rayTracingPipeline = VK_TRUE,
            .rayTracingPipelineTraceRaysIndirect = enabled.indirectDispatch
        };
        pNext = &pipelineFeatures;
    }
    if (enabled.positionFetch) {
        posFetchFeatures = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_POSITION_FETCH_FEATURES_KHR,
            .pNext = pNext,
            .rayTracingPositionFetch = VK_TRUE
        };
        pNext = &posFetchFeatures;
    }
    if (enabled.hitObjects) {
        reorderFeatures = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_EXT,
            .pNext = pNext,
            .rayTracingInvocationReorder = VK_TRUE
        };
        pNext = &reorderFeatures;
    }

    return pNext;
}

void RayTracingExtension::finalize(const ContextHandle& context) {
    //query cached properties
    //if the corresponding extension is not enabled, the values will be garbage
    //but we will not use them in that case anyway
    VkPhysicalDeviceRayTracingInvocationReorderPropertiesEXT reorderProps{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_PROPERTIES_EXT
    };
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR pipelineProps{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR,
        .pNext = &reorderProps
    };
    VkPhysicalDeviceAccelerationStructurePropertiesKHR accProps{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR,
        .pNext = &pipelineProps
    };
    VkPhysicalDeviceProperties2 devProps{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &accProps
    };
    vkGetPhysicalDeviceProperties2(context->physicalDevice, &devProps);

    shaderGroupProperties = {
        .shaderGroupHandleSize = pipelineProps.shaderGroupHandleSize,
        .shaderGroupBaseAlignment = pipelineProps.shaderGroupBaseAlignment,
        .shaderGroupHandleAlignment = pipelineProps.shaderGroupHandleAlignment
    };
    minAccelerationStructureScratchOffsetAlignment =
        accProps.minAccelerationStructureScratchOffsetAlignment;

    //assemble properties
    //values in the structs are undefined if the extension is not supported
    if (enabled.query || enabled.pipeline) {
        props.maxGeometryCount = accProps.maxGeometryCount;
        props.maxInstanceCount = accProps.maxInstanceCount;
        props.maxPrimitiveCount = accProps.maxPrimitiveCount;
        props.maxAccelerationStructures = std::min(
            accProps.maxDescriptorSetAccelerationStructures,
            accProps.maxPerStageDescriptorAccelerationStructures
        );
    }
    if (enabled.pipeline) {
        props.maxRayRecursionDepth = pipelineProps.maxRayRecursionDepth;
        props.maxRayDispatchCount = pipelineProps.maxRayDispatchInvocationCount;
        //result.maxRayHitAttributeSize = pipelineProps.maxRayHitAttributeSize;
        auto handleSize = pipelineProps.shaderGroupHandleSize;
        auto groupStride = pipelineProps.maxShaderGroupStride;
        props.maxShaderRecordSize = groupStride - handleSize;
    }
    if (enabled.hitObjects) {
        props.canReorder =
            VK_RAY_TRACING_INVOCATION_REORDER_MODE_REORDER_EXT ==
            reorderProps.rayTracingInvocationReorderReorderingHint;
    }
}

RayTracingExtension::RayTracingExtension(RayTracingFeatures features)
    : enabled(features)
    , props({})
    , shaderGroupProperties({})
    , minAccelerationStructureScratchOffsetAlignment(0)
    , reorderFeatures({ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_EXT })
    , posFetchFeatures({ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_POSITION_FETCH_FEATURES_KHR })
    , pipelineFeatures({ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR })
    , queryFeatures({ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR })
    , accStructFeatures({ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR })
{
    //indirect dispatch and invocation reorder imply pipeline
    if (enabled.indirectDispatch || enabled.hitObjects)
        enabled.pipeline = true;

    //required extensions
    extensions = {
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME
    };
    nExtension = 2;
    if (enabled.query)
        extensions[nExtension++] = VK_KHR_RAY_QUERY_EXTENSION_NAME;
    if (enabled.pipeline)
        extensions[nExtension++] = VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME;
    if (enabled.positionFetch)
        extensions[nExtension++] = VK_KHR_RAY_TRACING_POSITION_FETCH_EXTENSION_NAME;
    if (enabled.hitObjects)
        extensions[nExtension++] = VK_EXT_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME;
}
RayTracingExtension::~RayTracingExtension() = default;

}
