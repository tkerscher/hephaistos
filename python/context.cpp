#include "context.hpp"

#include <stdexcept>

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

//Keep device handles alive
//(ids are not guaranteed to be const between enumerations)
std::vector<hp::DeviceHandle> handles{};
std::vector<hp::DeviceInfo> devices{};
const std::vector<hp::DeviceInfo>& getDevices() {
    if (devices.empty()) {
        handles = hp::enumerateDevices();
        for (auto& h : handles)
            devices.push_back(hp::getDeviceInfo(h));
    }

    return devices;
}

void selectDevice(uint32_t id, bool force) {
    //check if we have to recreate context
    if (currentContext) {
        if (force) {
            //delete old context
            //objects are now undefined
            currentContext.reset();
            handles.clear();
            devices.clear();
            //instance should have been deleted now (doesn't really matter)
        }
        else {
            throw std::runtime_error("Cannot change device: There is already an active context! Specify force=True if you want to invalidate and replace active one.");
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
            currentContext = hp::createContext();
        }
        else if(selectedDeviceId < handles.size()) {
            currentContext = hp::createContext(handles[selectedDeviceId]);
        }
        else {
            throw std::runtime_error("There is no device with the selected id!");
        }
    }

    return currentContext;
}

void registerContextModule(nb::module_& m) {
    m.def("isVulkanAvailable", &hp::isVulkanAvailable, "Returns True if Vulkan is available on this system.");

    nb::class_<hp::DeviceInfo>(m, "Device")
        .def_ro("name", &hp::DeviceInfo::name)
        .def_ro("isDiscrete", &hp::DeviceInfo::isDiscrete);
    m.def("enumerateDevices", &getDevices, "Returns a list of all supported installed devices.");
    m.def("getCurrentDevice", []() { return hp::getDeviceInfo(getCurrentContext()); }, "Returns the currently active device. Note that this may initialize the context.");
    m.def("selectDevice", &selectDevice, "id"_a, "force"_a = false, "Sets the device on which the context will be initialized. Set force=True if an existing context should be destroyed.");
}
