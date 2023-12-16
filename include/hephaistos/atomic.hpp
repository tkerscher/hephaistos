#pragma once

#include "hephaistos/context.hpp"

namespace hephaistos {

/**
 * @brief List of atomic operations
*/
struct AtomicsProperties {
    bool bufferInt64Atomics;

    bool bufferFloat16Atomics;
    bool bufferFloat16AtomicAdd;
    bool bufferFloat16AtomicMinMax;

    bool bufferFloat32Atomics;
    bool bufferFloat32AtomicAdd;
    bool bufferFloat32AtomicMinMax;

    bool bufferFloat64Atomics;
    bool bufferFloat64AtomicAdd;
    bool bufferFloat64AtomicMinMax;

    bool sharedInt64Atomics;

    bool sharedFloat16Atomics;
    bool sharedFloat16AtomicAdd;
    bool sharedFloat16AtomicMinMax;

    bool sharedFloat32Atomics;
    bool sharedFloat32AtomicAdd;
    bool sharedFloat32AtomicMinMax;

    bool sharedFloat64Atomics;
    bool sharedFloat64AtomicAdd;
    bool sharedFloat64AtomicMinMax;

    bool imageInt64Atomics;

    bool imageFloat32Atomics;
    bool imageFloat32AtomicAdd;
    bool imageFloat32AtomicMinMax;
};

/**
 * @brief Returns the supported atomic operations on the given device
 * 
 * @param device Device to query
*/
[[nodiscard]]
HEPHAISTOS_API AtomicsProperties getAtomicsProperties(const DeviceHandle& device);
/**
 * @brief Returns the enabled atomic operations in the given context
 * 
 * @param context Context to query
*/
[[nodiscard]]
HEPHAISTOS_API const AtomicsProperties& getEnabledAtomics(const ContextHandle& context);

/**
 * @brief Creates an extension to enable the specified atomic operations
*/
[[nodiscard]]
HEPHAISTOS_API ExtensionHandle createAtomicsExtension(const AtomicsProperties& properties);

}
