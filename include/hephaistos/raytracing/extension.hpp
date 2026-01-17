#pragma once

#include "hephaistos/context.hpp"

namespace hephaistos {

/**
 * @brief List of optional ray tracing features.
 */
struct RayTracingFeatures {
    /**
     * @brief Support for ray queries
     */
    bool query;
    /**
     * @brief Support for ray tracing pipelines
     */
    bool pipeline;
    /**
     * @brief Support for indirect ray tracing dispatch
     */
    bool indirectDispatch;
    /**
     * @brief Support for fetching intersection position in shaders
     */
    bool positionFetch;
    /**
     * @brief Support for hit objects and shader invocation reorder
     */
    bool hitObjects;
};

/**
 * @brief Additional device properties specific to ray tracing
 */
struct RayTracingProperties {
    uint64_t maxGeometryCount;
    uint64_t maxInstanceCount;
    uint64_t maxPrimitiveCount;

    uint32_t maxAccelerationStructures;

    uint32_t maxRayRecursionDepth;
    uint32_t maxRayDispatchCount;
    // uint32_t maxRayHitAttributeSize;

    uint32_t maxShaderRecordSize;

    bool canReorder;
};

/**
 * @brief Queries the supported ray tracing features by the given device
 */
[[nodiscard]] HEPHAISTOS_API
RayTracingFeatures getRayTracingFeatures(const DeviceHandle& device);

/**
 * @brief Queries whether there is a device supporting the given ray tracing features.
 */
[[nodiscard]] HEPHAISTOS_API
bool isRayTracingSupported(RayTracingFeatures features);
/**
 * @brief Queries whether the given device supports the specific ray tracing features
 */
[[nodiscard]] HEPHAISTOS_API
bool isRayTracingSupported(const DeviceHandle& device, RayTracingFeatures features);

/**
 * @brief Queries the ray tracing properties of the given device
 */
[[nodiscard]] HEPHAISTOS_API
RayTracingProperties getRayTracingProperties(const DeviceHandle& device);
/**
 * @brief Queries the ray tracing properties of the device the given context belongs to
 */
[[nodiscard]] HEPHAISTOS_API
RayTracingProperties getCurrentRayTracingProperties(const ContextHandle& context);

/**
 * @brief Creates an extension handle enabling the given ray tracing features
 */
[[nodiscard]] HEPHAISTOS_API
ExtensionHandle createRayTracingExtension(RayTracingFeatures features);

/**
 * @brief Returns the enabled ray tracing features in the given context.
 */
[[nodiscard]] HEPHAISTOS_API
RayTracingFeatures getEnabledRayTracingFeatures(const ContextHandle& context);

}
