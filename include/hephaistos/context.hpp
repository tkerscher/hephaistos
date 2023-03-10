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

class HEPHAISTOS_API Resource {
public:
    const ContextHandle& getContext() const noexcept;

    Resource(const Resource&) = delete;
    Resource& operator=(const Resource&) = delete;

    virtual ~Resource();

protected:
    Resource(Resource&& other) noexcept;
    Resource& operator=(Resource&& other) noexcept;

    Resource(ContextHandle context);

private:
    ContextHandle context;
};

[[nodiscard]] HEPHAISTOS_API bool isVulkanAvailable();

[[nodiscard]] HEPHAISTOS_API std::vector<DeviceHandle> enumerateDevices();
[[nodiscard]] HEPHAISTOS_API DeviceHandle getDevice(const ContextHandle& context);
[[nodiscard]] HEPHAISTOS_API DeviceInfo getDeviceInfo(const DeviceHandle& device);
[[nodiscard]] HEPHAISTOS_API DeviceInfo getDeviceInfo(const ContextHandle& context);

[[nodiscard]] HEPHAISTOS_API ContextHandle createContext();
[[nodiscard]] HEPHAISTOS_API ContextHandle createContext(const DeviceHandle& device);

}
