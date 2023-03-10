#pragma once

#include <string>
#include <vector>

#include "hephaistos/config.hpp"
#include "hephaistos/handles.hpp"

namespace hephaistos {

struct DeviceInfo {
    std::string name;
    bool isDiscrete;
};

[[nodiscard]] HEPHAISTOS_API bool isVulkanAvailable();

[[nodiscard]] HEPHAISTOS_API std::vector<DeviceHandle> enumerateDevices();
[[nodiscard]] HEPHAISTOS_API DeviceHandle getDevice(const ContextHandle& context);
[[nodiscard]] HEPHAISTOS_API DeviceInfo getDeviceInfo(const DeviceHandle& device);
[[nodiscard]] HEPHAISTOS_API DeviceInfo getDeviceInfo(const ContextHandle& context);

[[nodiscard]] HEPHAISTOS_API ContextHandle createContext();
[[nodiscard]] HEPHAISTOS_API ContextHandle createContext(const DeviceHandle& device);

}
