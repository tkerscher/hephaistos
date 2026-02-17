#pragma once

#include <span>
#include <string>
#include <string_view>
#include <unordered_set>
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
    virtual void finalize(const ContextHandle& context) = 0;
    virtual ~Extension() = default;
};

/**
 * @brief Base class for resources allocated on a context
*/
class HEPHAISTOS_API Resource {
public:
    /**
     * @brief Checks whether the Resource is still alive, that is has not
     * been destroyed or moved.
    */
    operator bool() const noexcept;

    /**
     * @brief Returns the context this was created on. Null if destroyed.
    */
    const ContextHandle& getContext() const noexcept;

    /**
     * @brief Destroys the resource.
     * 
     * Destroys the resource without destroying the managing object.
    */
    void destroy();


    Resource(const Resource&) = delete;
    Resource& operator=(const Resource&) = delete;

    virtual ~Resource();

protected:
    /**
    * Destroy logic implemented by and specific to derived classes.
    */
    virtual void onDestroy() = 0;

    Resource(Resource&& other) noexcept;
    Resource& operator=(Resource&& other);

    Resource(ContextHandle context);

private:
    ContextHandle context;
};

#ifdef HEPHAISTOS_MANAGED_RESOURCES

/**
 * Takes a snapshot of currently alive resources and allows to destroy newly
 * created ones.
*/
class ResourceSnapshot {
public:
    /**
     * @brief Returns the number of newly created resources
    */
    [[nodiscard]] size_t count() const noexcept;

    /**
     * @brief Takes a snapshot of currently alive resources
     * 
     * @brief Fails if context has been destroyed
    */
    void capture();
    /**
     * @brief Destroys resources created since the last capture
     * 
     * @note Does nothing if context has been destroyed
    */
    void restore();

    ResourceSnapshot(const ContextHandle& context);
private:
    std::weak_ptr<vulkan::Context> context;
    std::unordered_set<Resource*> snapshot;
};

/**
 * @brief Returns the number of currently alive resources in the given context
 * 
 * @note Returns zero, if context is null
*/
[[nodiscard]] HEPHAISTOS_API size_t getResourceCount(const ContextHandle& context) noexcept;

/**
 * @brief Destroys all resources of the provided context
 * 
 * @note Does nothing if context is null
*/
HEPHAISTOS_API void destroyAllResources(const ContextHandle& context);

#endif

/**
 * @brief Queries vulkan support on this system
*/
[[nodiscard]] HEPHAISTOS_API bool isVulkanAvailable();
/**
 * @brief Queries extension support on the system
 * 
 * Checks whether there is any suitable device installed in the system
 * supporting the given extensions.
 * 
 * @params extensions List of extensions to query support for
 * @return True, if there is any device installed supporting all given extension,
 *         false otherwise.
*/
[[nodiscard]] HEPHAISTOS_API bool isDeviceSuitable(
    std::span<const ExtensionHandle> extension = {});
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

/**
 * @brief Checks the device health
 * 
 * Checks whether the context and the corresponding device are still in a
 * healthy state by performing a small task with known result. Success does
 * not guarantee that the device is actually healthy, but is a strong indicator.
 * Throws if device is unhealthy.
*/
HEPHAISTOS_API void checkDeviceHealth(const ContextHandle& context);

}
