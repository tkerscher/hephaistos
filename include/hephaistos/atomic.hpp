#pragma once

#include "hephaistos/context.hpp"

namespace hephaistos {

/**
 * @brief List of atomic operations
 * 
 * Structure containing boolean flags for various atomic operations. They are
 * name using the following scheme:
 * [memory][type]Atomic[ops]
 * 
 * [memory] describes the type of memory the data is stored and can have the
 * following values:
 *  - buffer: Uniform and Storage Buffers (including device buffer address)
 *  - shared: Shared memory
 *  - image: Images
 * 
 * [type] describes the data type and can have the following values:
 *  - Int64: Both signed and unsigned 64 bit integers
 *  - Float16: 16 bit floats (half precision)
 *  - Float32: 32 bit floats (single precision)
 *  - Float64: 64 bit floats (double precision)
 * 
 * [ops] describes the supported atomic operations and can have the following
 * values:
 *  - [None]: load, store, exchange, compSwap, add, min, max
 *  - [LoadStore]: load, store, exchange (notably no compSwap)
 *  - [Add]: add
 *  - [MinMax]: min, max
*/
struct AtomicsProperties {
    bool bufferInt64Atomics;

    bool bufferFloat16AtomicLoadStore;
    bool bufferFloat16AtomicAdd;
    bool bufferFloat16AtomicMinMax;

    bool bufferFloat32AtomicLoadStore;
    bool bufferFloat32AtomicAdd;
    bool bufferFloat32AtomicMinMax;

    bool bufferFloat64Atomics;
    bool bufferFloat64AtomicAdd;
    bool bufferFloat64AtomicMinMax;

    bool sharedInt64Atomics;

    bool sharedFloat16AtomicLoadStore;
    bool sharedFloat16AtomicAdd;
    bool sharedFloat16AtomicMinMax;

    bool sharedFloat32AtomicLoadStore;
    bool sharedFloat32AtomicAdd;
    bool sharedFloat32AtomicMinMax;

    bool sharedFloat64Atomics;
    bool sharedFloat64AtomicAdd;
    bool sharedFloat64AtomicMinMax;

    bool imageInt64Atomics;

    bool imageFloat32AtomicLoadStore;
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
