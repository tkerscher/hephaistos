#pragma once

#include <span>

#include "hephaistos/context.hpp"
#include "hephaistos/handles.hpp"

struct VkWriteDescriptorSet;

namespace hephaistos {

[[nodiscard]] HEPHAISTOS_API bool isRaytracingSupported(const DeviceHandle& device);
[[nodiscard]] HEPHAISTOS_API bool isRaytracingEnabled(const ContextHandle& context);

[[nodiscard]] HEPHAISTOS_API ExtensionHandle createRaytracingExtension();

}
