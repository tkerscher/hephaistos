#include "context.hpp"

#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <string>
#include <string_view>
#include <optional>
#include <unordered_set>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include <hephaistos/context.hpp>

namespace hp = hephaistos;
namespace nb = nanobind;
using namespace nb::literals;

namespace {

//context singleton
void _emptyDtor(hp::vulkan::Context*) {}
hp::ContextHandle currentContext{ nullptr, _emptyDtor };

//we select device via id, MaxId means let hephaistos decide
constexpr uint32_t MaxId = 0xFFFFFFFF;
uint32_t selectedDeviceId = MaxId;

//extension list
std::vector<hp::ExtensionHandle> extensions{};
std::vector<std::string_view> extension_names{};

//Keep device handles alive
//(ids are not guaranteed to be const between enumerations)
std::vector<hp::DeviceHandle> handles{};
std::vector<hp::DeviceInfo> enumerateDevices() {
    auto& devices = getDevices();
    std::vector<hp::DeviceInfo> result;
    for (auto& d : devices)
        result.push_back(hp::getDeviceInfo(d));
    return result;
}

void resetContext() {
    hp::destroyAllResources(currentContext);
    currentContext.reset();
    //handles are still valid, so we keep them
    //note that this keeps also the vulkan instance alive
    //handles.clear();
}

void selectDevice(uint32_t id, bool force) {
    //check if we have to recreate context
    if (currentContext) {
        if (force) {
            resetContext();            
        }
        else {
            throw std::runtime_error(
                "Cannot change device: There is already an active context! Specify "
                "force=True if you want to invalidate and replace active one.");
        }
    }

    //range check
    if (id >= getDevices().size())
        throw std::runtime_error("There is no device with the selected id!");
    
    //set id
    selectedDeviceId = id;
}

} //end namespace

const hp::ContextHandle& getCurrentContext() {
    if (!currentContext) {
        if (selectedDeviceId == MaxId) {
            currentContext = hp::createContext(extensions);
        }
        else if(selectedDeviceId < handles.size()) {
            currentContext = hp::createContext(handles[selectedDeviceId], extensions);
        }
        else {
            throw std::runtime_error("There is no device with the selected id!");
        }
    }

    return currentContext;
}

const std::vector<hp::DeviceHandle>& getDevices() {
    if (handles.empty())
        handles = hp::enumerateDevices();

    return handles;
}

void addExtension(hp::ExtensionHandle extension, bool force) {
    //check if the extension was already added
    auto name = extension->getExtensionName();
    auto& reg = extension_names;
    if (std::find(reg.begin(), reg.end(), name) != reg.end())
        return;

    //check if we have to recreate context
    if (currentContext) {
        if (force) {
            resetContext();
        }
        else {
            throw std::runtime_error(
                "Cannot change device: There is already an active context! Specify "
                "force=True if you want to invalidate and replace active one.");
        }
    }

    //add extension
    extension_names.push_back(name);
    extensions.push_back(std::move(extension));
}

void registerContextModule(nb::module_& m) {
    m.def("isVulkanAvailable", &hp::isVulkanAvailable,
        "Returns True if Vulkan is available on this system.");

    nb::class_<hp::DeviceInfo>(m, "Device",
            "Handle for a physical device implementing the Vulkan API. "
            "Contains basic properties of the device.")
        .def_ro("name", &hp::DeviceInfo::name, "Name of the device")
        .def_ro("isDiscrete", &hp::DeviceInfo::isDiscrete,
            "True, if the device is a discrete GPU. "
            "Can be useful to distinguish from integrated ones.")
        .def("__repr__", [](const hp::DeviceInfo& info){
            std::ostringstream str;
            str << info.name;
            if (info.isDiscrete)
                str << " (discrete)";
            return str.str();
        });
    m.def("enumerateDevices", &enumerateDevices,
        "Returns a list of all supported installed devices.");
    m.def("getCurrentDevice", []() { return hp::getDeviceInfo(getCurrentContext()); },
        "Returns the currently active device. Note that this may initialize the context.");
    m.def("selectDevice", &selectDevice, "id"_a, "force"_a = false,
        "Sets the device on which the context will be initialized. "
        "Set force=True if an existing context should be destroyed.");
    
    m.def("isDeviceSuitable", [](uint32_t id) {
        //range check
        if (id >= getDevices().size())
            throw std::runtime_error("There is no device with the selected id!");
        
        return hp::isDeviceSuitable(getDevices()[id], extensions);
    }, "Returns True if the device given by its id supports all enabled extensions");
    m.def("suitableDeviceAvailable", []() {
            //check all devices
            for (auto& device : getDevices()) {
                if (hp::isDeviceSuitable(device, extensions))
                    return true;
            }
            //no suitable device found
            return false;
        },
        "Returns True, if there is a device available supporting all enabled extensions");
    
    m.def("destroyResources", []() { hp::destroyAllResources(getCurrentContext()); },
        "Destroys all resources on the GPU and frees their allocated memory");
    m.def("getResourceCount", []() { return hp::getResourceCount(getCurrentContext()); },
        "Counts the number of resources currently alive");
    
    nb::class_<hp::ResourceSnapshot>(m, "ResourceSnapshot",
        "Takes a snapshot of currently alive resources and allows to destroy "
        "resources created afterwards.")
        .def("__init__", [](hp::ResourceSnapshot* snap) {
            new (snap) hp::ResourceSnapshot(getCurrentContext());
        })
        .def_prop_ro("count", &hp::ResourceSnapshot::count,
            "Number of resources created while context was activated")
        .def("capture", &hp::ResourceSnapshot::capture,
            "Takes a snapshot of currently alive resources")
        .def("restore", &hp::ResourceSnapshot::restore,
            "Destroys resources created since the last capture")
        .def("__enter__", &hp::ResourceSnapshot::capture)
        .def("__exit__",
            [](
                hp::ResourceSnapshot& snap,
                const std::optional<nb::object>& exc_type,
                const std::optional<nb::object>& exc_value,
                const std::optional<nb::object>& traceback
            ) {
                snap.restore();
            },
            "exc_type"_a.none(), "exc_value"_a.none(), "traceback"_a.none());
}
