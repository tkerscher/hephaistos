#include <nanobind/nanobind.h>

#include <algorithm>
#include <stdexcept>

#include <hephaistos/program.hpp>
#include "context.hpp"

namespace hp = hephaistos;
namespace nb = nanobind;
using namespace nb::literals;

extern nb::class_<hp::Command> commandClass;

namespace {

hp::SubgroupProperties getSubgroupProperties(uint32_t i) {
    auto& devices = getDevices();
    if (i >= devices.size())
        throw std::runtime_error("There is no device with the given id!");
    
    return hp::getSubgroupProperties(devices[i]);
}

}

void registerProgramModule(nb::module_& m) {
    nb::class_<hp::SubgroupProperties>(m, "SubgroupProperties");
    m.def("getSubgroupProperties",
        []() -> hp::SubgroupProperties { return hp::getSubgroupProperties(getCurrentContext()); },
        "Returns the properties specific to subgroups (waves).");
    m.def("getSubgroupProperties", &getSubgroupProperties,
        "Returns the properties specific to subgroups (waves).");

    nb::class_<hp::DispatchCommand>(m, "DispatchCommand", commandClass)
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
        .def("dispatch",
                [](const hp::Program& p, uint32_t x, uint32_t y, uint32_t z) -> hp::DispatchCommand {
                    return p.dispatch(x, y, z); },
            "x"_a = 1, "y"_a = 1, "z"_a = 1,
            "Dispatches a program execution with the given workgroup size.")
        .def("dispatchPush",
            [](const hp::Program& p, nb::bytes push, uint32_t x, uint32_t y, uint32_t z)
                 -> hp::DispatchCommand {
                    return p.dispatch(
                        std::span<const std::byte>{ reinterpret_cast<const std::byte*>(push.c_str()), push.size() }, x, y, z); },
                "push"_a, "x"_a = 1, "y"_a = 1, "z"_a = 1,
                "Dispatches a program execution with the given push data and workgroup size.");
}
