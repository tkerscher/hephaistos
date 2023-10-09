#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <sstream>
#include <stdexcept>

#include <hephaistos/program.hpp>
#include "context.hpp"

namespace hp = hephaistos;
namespace nb = nanobind;
using namespace nb::literals;

namespace {

hp::SubgroupProperties getSubgroupProperties(uint32_t i) {
    auto& devices = getDevices();
    if (i >= devices.size())
        throw std::runtime_error("There is no device with the given id!");
    
    return hp::getSubgroupProperties(devices[i]);
}

}

void registerProgramModule(nb::module_& m) {
    nb::class_<hp::SubgroupProperties>(m, "SubgroupProperties")
        .def_ro("subgroupSize", &hp::SubgroupProperties::subgroupSize)
        .def_ro("basicSupport", &hp::SubgroupProperties::basicSupport)
        .def_ro("voteSupport", &hp::SubgroupProperties::voteSupport)
        .def_ro("arithmeticSupport", &hp::SubgroupProperties::arithmeticSupport)
        .def_ro("ballotSupport", &hp::SubgroupProperties::ballotSupport)
        .def_ro("shuffleSupport", &hp::SubgroupProperties::shuffleSupport)
        .def_ro("shuffleRelativeSupport", &hp::SubgroupProperties::shuffleRelativeSupport)
        .def_ro("shuffleClusteredSupport", &hp::SubgroupProperties::shuffleClusteredSupport)
        .def_ro("quadSupport", &hp::SubgroupProperties::quadSupport);
    m.def("getSubgroupProperties",
        []() -> hp::SubgroupProperties { return hp::getSubgroupProperties(getCurrentContext()); },
        "Returns the properties specific to subgroups (waves).");
    m.def("getSubgroupProperties", &getSubgroupProperties,
        "Returns the properties specific to subgroups (waves).");

    nb::enum_<hp::DescriptorType>(m, "DescriptorType")
        .value("COMBINED_IMAGE_SAMPLER", hp::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .value("STORAGE_IMAGE", hp::DescriptorType::STORAGE_IMAGE)
        .value("UNIFORM_BUFFER", hp::DescriptorType::UNIFORM_BUFFER)
        .value("STORAGE_BUFFER", hp::DescriptorType::STORAGE_BUFFER)
        .value("ACCELERATION_STRUCTURE", hp::DescriptorType::ACCELERATION_STRUCTURE);

    nb::class_<hp::ImageBindingTraits>(m, "ImageBindingTraits")
        .def_ro("format", &hp::ImageBindingTraits::format)
        .def_ro("dims", &hp::ImageBindingTraits::dims);
    
    nb::class_<hp::BindingTraits>(m, "BindingTraits")
        .def_ro("name", &hp::BindingTraits::name)
        .def_ro("binding", &hp::BindingTraits::binding)
        .def_ro("type", &hp::BindingTraits::type)
        .def_ro("imageTraits", &hp::BindingTraits::imageTraits)
        .def_ro("count", &hp::BindingTraits::count);

    nb::class_<hp::LocalSize>(m, "LocalSize")
        .def(nb::init<>())
        .def_rw("x", &hp::LocalSize::x)
        .def_rw("y", &hp::LocalSize::y)
        .def_rw("z", &hp::LocalSize::z)
        .def("__repr__", [](const hp::LocalSize& s){
            std::ostringstream str;
            str << "{ x: " << s.x << ", y: " << s.y << ", z: " << s.z << " }\n";
            return str.str();
        });

    nb::class_<hp::DispatchCommand, hp::Command>(m, "DispatchCommand")
        .def_rw("groupCountX", &hp::DispatchCommand::groupCountX)
        .def_rw("groupCountY", &hp::DispatchCommand::groupCountY)
        .def_rw("groupCountZ", &hp::DispatchCommand::groupCountZ);
    
    nb::class_<hp::Program>(m, "Program")
        .def("__init__", [](hp::Program* p, nb::bytes code) {
                new (p) hp::Program(getCurrentContext(),
                    std::span<const uint32_t>{ reinterpret_cast<const uint32_t*>(code.c_str()), code.size() / 4 });
            }, "code"_a)
        .def("__init__", [](hp::Program* p, nb::bytes code, nb::bytes spec) {
                new (p) hp::Program(
                    getCurrentContext(),
                    std::span<const uint32_t>{ reinterpret_cast<const uint32_t*>(code.c_str()), code.size() / 4 },
                    std::span<const std::byte>{ reinterpret_cast<const std::byte*>(spec.c_str()), spec.size() });
            }, "code"_a, "specialization"_a)
        .def_prop_ro("localSize", [](const hp::Program& p){ return p.getLocalSize(); }, "Returns the size of the local work group.")
        .def_prop_ro("bindings", [](const hp::Program& p){ return p.listBindings(); }, "Returns a list of all bindings.")
        .def("dispatch",
                [](const hp::Program& p, uint32_t x, uint32_t y, uint32_t z) -> hp::DispatchCommand {
                    return p.dispatch(x, y, z); },
            "x"_a = 1, "y"_a = 1, "z"_a = 1,
            "Dispatches a program execution with the given workgroup size.")
        .def("dispatchPush",
            [](const hp::Program& p, nb::bytes push, uint32_t x, uint32_t y, uint32_t z)
                 -> hp::DispatchCommand {
                    return p.dispatch(
                        std::span<const std::byte>{ reinterpret_cast<const std::byte*>(push.c_str()), push.size() },
                        x, y, z
                    );
                }, nb::keep_alive<0,2>(), //keep push bytes as long alive as the dispatch command
                "push"_a, "x"_a = 1, "y"_a = 1, "z"_a = 1,
                "Dispatches a program execution with the given push data and workgroup size.");
    
    nb::class_<hp::FlushMemoryCommand, hp::Command>(m, "FlushMemoryCommand")
        .def("__init__", [](hp::FlushMemoryCommand* cmd) { new (cmd) hp::FlushMemoryCommand(getCurrentContext()); });
    m.def("flushMemory", []() -> hp::FlushMemoryCommand { return hp::flushMemory(getCurrentContext()); },
        "Returns a command for flushing memory writes.");
}
