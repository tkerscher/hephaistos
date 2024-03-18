#pragma once

#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "hephaistos/config.hpp"
#include "hephaistos/handles.hpp"

namespace hephaistos {

/**
 * @brief Information about a device
*/
struct DeviceInfo {
    /**
     * @brief Name of the device
    */
    std::string name;
    /**
     * @brief Wether the device is a discrete GPU
    */
    bool isDiscrete;
};

/**
 * @brief Base class for extensions
 * 
 * Extensions provide extra features which support by a device can be queried
 * and optionally enabled during the creation of a context.
*/
class Extension {
public:
    /**
     * @brief Checks wether the extension is supported by the given device
     * 
     * @param device Device to query support
    */
    virtual bool isDeviceSupported(const DeviceHandle& device) const = 0;
    /**
     * @brief Returns name of the extension
    */
    virtual std::string_view getExtensionName() const = 0;

public: //internal
    virtual std::span<const char* const> getDeviceExtensions() const = 0;
    //we create our DeviceCreateInfo pnext chain in reverse order:
    //the first extension is called chain(nullptr) and returns the address
    //of its chain end. This is passed to the next extension until the last
    //pointer is stored in the DeviceCreateInfo pNext
    virtual void* chain(void* pNext) = 0;
    virtual ~Extension() = default;
};

/**
 * @brief Base class for resources allocated on a context
*/
class HEPHAISTOS_API Resource {
public:
    /**
     * @brief Returns the context this was created on
    */
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

/**
 * @brief Queries vulkan support on this system
*/
[[nodiscard]] HEPHAISTOS_API bool isVulkanAvailable();
/**
 * @brief Queries extension support by the given device
 * 
 * @param device Device to query support
 * @param extensions List of extensions to query support for
 * @return True, if all extensions are supported by the device, false otherwise
*/
[[nodiscard]] HEPHAISTOS_API bool isDeviceSuitable(
    const DeviceHandle& device, std::span<const ExtensionHandle> extensions = {});

/**
 * @brief Returns all supported devices
 * 
 * @note If a device does not show up, it might not met the minimum requirements 
*/
[[nodiscard]] HEPHAISTOS_API std::vector<DeviceHandle> enumerateDevices();
/**
 * @brief Returns the device on which the given context was created
*/
[[nodiscard]] HEPHAISTOS_API DeviceHandle getDevice(const ContextHandle& context);
/**
 * @brief Returns information about the given device
*/
[[nodiscard]] HEPHAISTOS_API DeviceInfo getDeviceInfo(const DeviceHandle& device);
/**
 * @brief Returns information about the device used to create the given context
*/
[[nodiscard]] HEPHAISTOS_API DeviceInfo getDeviceInfo(const ContextHandle& context);

/**
 * @brief Creates a new context
 * 
 * Creates a new context on a device that supports all demanded extensions.
 * Throws if no suitable device is available. Prefers discrete devices.
 * 
 * @param extensions Extensions to enable
 * @return Handle to newly created context
*/
[[nodiscard]] HEPHAISTOS_API ContextHandle createContext(
    std::span<ExtensionHandle> extensions = {});
/**
 * @brief Creates a new context on the given device
 * 
 * Creates a new context on the given device and enables the demanded extensions.
 * Throws if the given device does not support the extensions.
 * 
 * @param device Device on which to create the context
 * @param extensions Extensions to enable
 * @return Handle to newly created context
*/
[[nodiscard]] HEPHAISTOS_API ContextHandle createContext(
    const DeviceHandle& device, std::span<ExtensionHandle> extensions = {});

/**
 * @brief Creates an empty context
 * 
 * Empty context can be used for default initialize contexts, but does not
 * support creating any resources.
*/
[[nodiscard]] inline ContextHandle createEmptyContext() {
    return ContextHandle{ nullptr, [](vulkan::Context*) {} };
}

}
