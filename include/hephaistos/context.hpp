#pragma once

#include <span>
#include <string>
#include <vector>

#include "hephaistos/config.hpp"
#include "hephaistos/handles.hpp"

namespace hephaistos {

struct DeviceInfo {
    std::string name;
    bool isDiscrete;
};

class Extension {
public:
    virtual bool isDeviceSupported(const DeviceHandle& device) const = 0;
    virtual std::string_view getExtensionName() const = 0;
    virtual std::span<const char* const> getDeviceExtensions() const = 0;
    //we create our DeviceCreateInfo pnext chain in reverse order:
    //the first extension is called chain(nullptr) and returns the address
    //of its chain end. This is passed to the next extension until the last
    //pointer is stored in the DeviceCreateInfo pNext
    virtual void* chain(void* pNext) = 0;
    virtual ~Extension() = default;
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
[[nodiscard]] HEPHAISTOS_API bool isDeviceSuitable(
    const DeviceHandle& device, std::span<const ExtensionHandle> extensions = {});

[[nodiscard]] HEPHAISTOS_API std::vector<DeviceHandle> enumerateDevices();
[[nodiscard]] HEPHAISTOS_API DeviceHandle getDevice(const ContextHandle& context);
[[nodiscard]] HEPHAISTOS_API DeviceInfo getDeviceInfo(const DeviceHandle& device);
[[nodiscard]] HEPHAISTOS_API DeviceInfo getDeviceInfo(const ContextHandle& context);

[[nodiscard]] HEPHAISTOS_API ContextHandle createContext(
    std::span<ExtensionHandle> extensions = {});
[[nodiscard]] HEPHAISTOS_API ContextHandle createContext(
    const DeviceHandle& device, std::span<ExtensionHandle> extensions = {});

[[nodiscard]] inline ContextHandle createEmptyContext() {
    return ContextHandle{ nullptr, [](vulkan::Context*) {} };
}

}
