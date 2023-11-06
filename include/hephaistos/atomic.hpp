#pragma once

#include "hephaistos/context.hpp"

namespace hephaistos {

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

[[nodiscard]]
HEPHAISTOS_API AtomicsProperties getAtomicsProperties(const DeviceHandle& device);
[[nodiscard]]
HEPHAISTOS_API const AtomicsProperties& getEnabledAtomics(const ContextHandle& context);

[[nodiscard]]
HEPHAISTOS_API ExtensionHandle createAtomicsExtension(const AtomicsProperties& properties);

}
