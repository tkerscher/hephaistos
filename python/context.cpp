#include "context.hpp"

#include <stdexcept>
#include <sstream>
#include <string>
#include <string_view>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
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

void selectDevice(uint32_t id, bool force) {
    //check if we have to recreate context
    if (currentContext) {
        if (force) {
            //delete old context
            //objects are now undefined
            currentContext.reset();
            handles.clear();
            //instance should have been deleted now (doesn't really matter)
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

}

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
            //delete old context
            //objects are now undefined
            currentContext.reset();
            handles.clear();
            //instance should have been deleted now (doesn't really matter)
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
}
