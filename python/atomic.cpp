#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>

#include <sstream>

#include <hephaistos/atomic.hpp>

#include "context.hpp"

namespace hp = hephaistos;
namespace nb = nanobind;
using namespace nb::literals;

void registerAtomicModule(nb::module_& m) {
    nb::class_<hp::AtomicsProperties>(m, "AtomicsProperties",
            "List of atomic functions a device supports or are enabled")
        .def_ro("bufferInt64Atomics", &hp::AtomicsProperties::bufferInt64Atomics)
        .def_ro("bufferFloat16Atomics", &hp::AtomicsProperties::bufferFloat16Atomics)
        .def_ro("bufferFloat16AtomicAdd", &hp::AtomicsProperties::bufferFloat16AtomicAdd)
        .def_ro("bufferFloat16AtomicMinMax", &hp::AtomicsProperties::bufferFloat16AtomicMinMax)
        .def_ro("bufferFloat32Atomics", &hp::AtomicsProperties::bufferFloat32Atomics)
        .def_ro("bufferFloat32AtomicAdd", &hp::AtomicsProperties::bufferFloat32AtomicAdd)
        .def_ro("bufferFloat32AtomicMinMax", &hp::AtomicsProperties::bufferFloat32AtomicMinMax)
        .def_ro("bufferFloat64Atomics", &hp::AtomicsProperties::bufferFloat64Atomics)
        .def_ro("bufferFloat64AtomicAdd", &hp::AtomicsProperties::bufferFloat64AtomicAdd)
        .def_ro("bufferFloat64AtomicMinMax", &hp::AtomicsProperties::bufferFloat64AtomicMinMax)
        .def_ro("sharedInt64Atomics", &hp::AtomicsProperties::sharedInt64Atomics)
        .def_ro("sharedFloat16Atomics", &hp::AtomicsProperties::sharedFloat16Atomics)
        .def_ro("sharedFloat16AtomicAdd", &hp::AtomicsProperties::sharedFloat16AtomicAdd)
        .def_ro("sharedFloat16AtomicMinMax", &hp::AtomicsProperties::sharedFloat16AtomicMinMax)
        .def_ro("sharedFloat32Atomics", &hp::AtomicsProperties::sharedFloat32Atomics)
        .def_ro("sharedFloat32AtomicAdd", &hp::AtomicsProperties::sharedFloat32AtomicAdd)
        .def_ro("sharedFloat32AtomicMinMax", &hp::AtomicsProperties::sharedFloat32AtomicMinMax)
        .def_ro("sharedFloat64Atomics", &hp::AtomicsProperties::sharedFloat64Atomics)
        .def_ro("sharedFloat64AtomicAdd", &hp::AtomicsProperties::sharedFloat64AtomicAdd)
        .def_ro("sharedFloat64AtomicMinMax", &hp::AtomicsProperties::sharedFloat64AtomicMinMax)
        .def_ro("imageInt64Atomics", &hp::AtomicsProperties::imageInt64Atomics)
        .def_ro("imageFloat32Atomics", &hp::AtomicsProperties::imageFloat32Atomics)
        .def_ro("imageFloat32AtomicAdd", &hp::AtomicsProperties::imageFloat32AtomicAdd)
        .def_ro("imageFloat32AtomicMinMax", &hp::AtomicsProperties::imageFloat32AtomicMinMax)
        .def("__repr__", [](const hp::AtomicsProperties& p){
            std::ostringstream str;
            str << std::boolalpha;
            str << "bufferInt64Atomics:        " << !!p.bufferInt64Atomics << '\n';
            str << "bufferFloat16Atomics:      " << !!p.bufferFloat16Atomics << '\n';
            str << "bufferFloat16AtomicAdd:    " << !!p.bufferFloat16AtomicAdd << '\n';
            str << "bufferFloat16AtomicMinMax: " << !!p.bufferFloat16AtomicMinMax << '\n';
            str << "bufferFloat32Atomics:      " << !!p.bufferFloat32Atomics << '\n';
            str << "bufferFloat32AtomicAdd:    " << !!p.bufferFloat32AtomicAdd << '\n';
            str << "bufferFloat32AtomicMinMax: " << !!p.bufferFloat32AtomicMinMax << '\n';
            str << "bufferFloat64Atomics:      " << !!p.bufferFloat64Atomics << '\n';
            str << "bufferFloat64AtomicAdd:    " << !!p.bufferFloat64AtomicAdd << '\n';
            str << "bufferFloat64AtomicMinMax: " << !!p.bufferFloat64AtomicMinMax << '\n';
            str << "sharedInt64Atomics:        " << !!p.sharedInt64Atomics << '\n';
            str << "sharedFloat16Atomics:      " << !!p.sharedFloat16Atomics << '\n';
            str << "sharedFloat16AtomicAdd:    " << !!p.sharedFloat16AtomicAdd << '\n';
            str << "sharedFloat16AtomicMinMax: " << !!p.sharedFloat16AtomicMinMax << '\n';
            str << "sharedFloat32Atomics:      " << !!p.sharedFloat32Atomics << '\n';
            str << "sharedFloat32AtomicAdd:    " << !!p.sharedFloat32AtomicAdd << '\n';
            str << "sharedFloat32AtomicMinMax: " << !!p.sharedFloat32AtomicMinMax << '\n';
            str << "sharedFloat64Atomics:      " << !!p.sharedFloat64Atomics << '\n';
            str << "sharedFloat64AtomicAdd:    " << !!p.sharedFloat64AtomicAdd << '\n';
            str << "sharedFloat64AtomicMinMax: " << !!p.sharedFloat64AtomicMinMax << '\n';
            str << "imageInt64Atomics:         " << !!p.imageInt64Atomics << '\n';
            str << "imageFloat32Atomics:       " << !!p.imageFloat32Atomics << '\n';
            str << "imageFloat32AtomicAdd:     " << !!p.imageFloat32AtomicAdd << '\n';
            str << "imageFloat32AtomicMinMax:  " << !!p.imageFloat32AtomicMinMax;
            return str.str();
        });
    
    m.def("getAtomicsProperties", [](uint32_t id) -> hp::AtomicsProperties {
        auto& devices = getDevices();
        if (id >= devices.size())
            throw std::runtime_error("There is no device with the selected id!");
        return hp::getAtomicsProperties(devices[id]);
    }, "id"_a, "Returns the atomic capabilities of the device given by its id");
    m.def("getEnabledAtomics", []() -> hp::AtomicsProperties {
        return hp::getEnabledAtomics(getCurrentContext());
    }, "Returns the in the current context enabled atomic features");

    m.def("enableAtomics", [](const nb::set& flags, bool force) {
        //build atomic properties
        hp::AtomicsProperties props{
            .bufferInt64Atomics = flags.contains("bufferInt64Atomics"),
            .bufferFloat16Atomics = flags.contains("bufferFloat16Atomics"),
            .bufferFloat16AtomicAdd = flags.contains("bufferFloat16AtomicAdd"),
            .bufferFloat16AtomicMinMax = flags.contains("bufferFloat16AtomicMinMax"),
            .bufferFloat32Atomics = flags.contains("bufferFloat32Atomics"),
            .bufferFloat32AtomicAdd = flags.contains("bufferFloat32AtomicAdd"),
            .bufferFloat32AtomicMinMax = flags.contains("bufferFloat32AtomicMinMax"),
            .bufferFloat64Atomics = flags.contains("bufferFloat64Atomics"),
            .bufferFloat64AtomicAdd = flags.contains("bufferFloat64AtomicAdd"),
            .bufferFloat64AtomicMinMax = flags.contains("bufferFloat64AtomicMinMax"),
            .sharedInt64Atomics = flags.contains("sharedInt64Atomics"),
            .sharedFloat16Atomics = flags.contains("sharedFloat16Atomics"),
            .sharedFloat16AtomicAdd = flags.contains("sharedFloat16AtomicAdd"),
            .sharedFloat16AtomicMinMax = flags.contains("sharedFloat16AtomicMinMax"),
            .sharedFloat32Atomics = flags.contains("sharedFloat32Atomics"),
            .sharedFloat32AtomicAdd = flags.contains("sharedFloat32AtomicAdd"),
            .sharedFloat32AtomicMinMax = flags.contains("sharedFloat32AtomicMinMax"),
            .sharedFloat64Atomics = flags.contains("sharedFloat64Atomics"),
            .sharedFloat64AtomicAdd = flags.contains("sharedFloat64AtomicAdd"),
            .sharedFloat64AtomicMinMax = flags.contains("sharedFloat64AtomicMinMax"),
            .imageInt64Atomics = flags.contains("imageInt64Atomics"),
            .imageFloat32Atomics = flags.contains("imageFloat32Atomics"),
            .imageFloat32AtomicAdd = flags.contains("imageFloat32AtomicAdd"),
            .imageFloat32AtomicMinMax = flags.contains("imageFloat32AtomicMinMax")
        };
        //add extension
        addExtension(hp::createAtomicsExtension(props), force);
    }, "flags"_a, "force"_a = false,
    "Enables the atomic features contained in the given set by their name. "
    "Set force=True if an existing context should be destroyed.");
}
