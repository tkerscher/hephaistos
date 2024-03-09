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

void printBindingType(std::ostringstream& str, hp::ParameterType type) {
    switch(type) {
    case hp::ParameterType::COMBINED_IMAGE_SAMPLER:
        str << "COMBINED_IMAGE_SAMPLER";
        break;
    case hp::ParameterType::STORAGE_IMAGE:
        str << "STORAGE_IMAGE";
        break;
    case hp::ParameterType::UNIFORM_BUFFER:
        str << "UNIFORM_BUFFER";
        break;
    case hp::ParameterType::STORAGE_BUFFER:
        str << "STORAGE_BUFFER";
        break;
    case hp::ParameterType::ACCELERATION_STRUCTURE:
        str << "ACCELERATION_STRUCTURE";
        break;
    default:
        str << "UNKNOWN";
        break;
    }
}

void printBinding(std::ostringstream& str, const hp::BindingTraits& b) {
    str << b.binding << ": " << b.name << " (";
    printBindingType(str, b.type);
    if (b.count > 1)
        str << '[' << b.count << ']';
    str << ")";
}

}

void registerProgramModule(nb::module_& m) {
    nb::class_<hp::SubgroupProperties>(m, "SubgroupProperties",
            "List of subgroup properties and supported operations")
        .def_ro("subgroupSize", &hp::SubgroupProperties::subgroupSize,
            "Threads per subgroup")
        .def_ro("basicSupport", &hp::SubgroupProperties::basicSupport,
            "Support for GL_KHR_shader_subgroup_basic")
        .def_ro("voteSupport", &hp::SubgroupProperties::voteSupport,
            "Support for GL_KHR_shader_subgroup_vote")
        .def_ro("arithmeticSupport", &hp::SubgroupProperties::arithmeticSupport,
            "Support for GL_KHR_shader_subgroup_arithmetic")
        .def_ro("ballotSupport", &hp::SubgroupProperties::ballotSupport,
            "Support for GL_KHR_shader_subgroup_ballot")
        .def_ro("shuffleSupport", &hp::SubgroupProperties::shuffleSupport,
            "Support for GL_KHR_shader_subgroup_shuffle")
        .def_ro("shuffleRelativeSupport", &hp::SubgroupProperties::shuffleRelativeSupport,
            "Support for GL_KHR_shader_subgroup_shuffle_relative")
        .def_ro("shuffleClusteredSupport", &hp::SubgroupProperties::shuffleClusteredSupport,
            "Support for GL_KHR_shader_subgroup_clustered")
        .def_ro("quadSupport", &hp::SubgroupProperties::quadSupport,
            "Support for GL_KHR_shader_subgroup_quad")
        .def_ro("uniformControlFlowSupport", &hp::SubgroupProperties::uniformControlFlowSupport,
            "Support for GL_EXT_subgroup_uniform_control_flow")
        .def_ro("maximalReconvergenceSupport", &hp::SubgroupProperties::maximalReconvergenceSupport,
            "Support for GL_EXT_maximal_reconvergence")
        .def("__repr__", [](const hp::SubgroupProperties& props) {
            std::ostringstream str;
            str << "subgroupSize:            " << props.subgroupSize << '\n';
            str << "basicSupport:            " << props.basicSupport << '\n';
            str << "voteSupport:             " << props.voteSupport << '\n';
            str << "arithmeticSupport:       " << props.arithmeticSupport << '\n';
            str << "ballotSupport:           " << props.ballotSupport << '\n';
            str << "shuffleSupport:          " << props.shuffleSupport << '\n';
            str << "shuffleRelativeSupport:  " << props.shuffleRelativeSupport << '\n';
            str << "shuffleClusteredSupport: " << props.shuffleClusteredSupport << '\n';
            str << "quadSupport:             " << props.quadSupport << '\n';
            str << "uniformControlFlow:      " << props.uniformControlFlowSupport << '\n';
            str << "maximalReconvergence:    " << props.maximalReconvergenceSupport;
            
            return str.str();
        });
    m.def("getSubgroupProperties",
        []() -> hp::SubgroupProperties { return hp::getSubgroupProperties(getCurrentContext()); },
        "Returns the properties specific to subgroups (waves).");
    m.def("getSubgroupProperties", &getSubgroupProperties,
        "Returns the properties specific to subgroups (waves).");

    nb::enum_<hp::ParameterType>(m, "ParameterType", "Type of parameter")
        .value("COMBINED_IMAGE_SAMPLER", hp::ParameterType::COMBINED_IMAGE_SAMPLER)
        .value("STORAGE_IMAGE", hp::ParameterType::STORAGE_IMAGE)
        .value("UNIFORM_BUFFER", hp::ParameterType::UNIFORM_BUFFER)
        .value("STORAGE_BUFFER", hp::ParameterType::STORAGE_BUFFER)
        .value("ACCELERATION_STRUCTURE", hp::ParameterType::ACCELERATION_STRUCTURE);

    nb::class_<hp::ImageBindingTraits>(m, "ImageBindingTraits",
            "Properties a binding expects from a bound image")
        .def_ro("format", &hp::ImageBindingTraits::format, "Expected image format")
        .def_ro("dims", &hp::ImageBindingTraits::dims, "Image dimensions");
    
    nb::class_<hp::BindingTraits>(m, "BindingTraits",
            "Properties of binding found in programs")
        .def_ro("name", &hp::BindingTraits::name, "Name of the binding. Might be empty.")
        .def_ro("binding", &hp::BindingTraits::binding, "Index of the binding")
        .def_ro("type", &hp::BindingTraits::type, "Type of the binding")
        .def_ro("imageTraits", &hp::BindingTraits::imageTraits,
            "Properties of the image if one is expected")
        .def_ro("count", &hp::BindingTraits::count,
            "Number of elements in binding, i.e. array size")
        .def("__repr__", [](const hp::BindingTraits& b){
            std::ostringstream str;
            printBinding(str, b);
            return str.str();
        });

    nb::class_<hp::LocalSize>(m, "LocalSize",
            "Description if the local size, i.e. the number and arrangement of "
            "threads in a single thread group")
        .def(nb::init<>())
        .def_rw("x", &hp::LocalSize::x, "Number of threads in X dimension")
        .def_rw("y", &hp::LocalSize::y, "Number of threads in Y dimension")
        .def_rw("z", &hp::LocalSize::z, "Number of threads in Z dimension")
        .def("__repr__", [](const hp::LocalSize& s){
            std::ostringstream str;
            str << "{ x: " << s.x << ", y: " << s.y << ", z: " << s.z << " }\n";
            return str.str();
        });

    nb::class_<hp::DispatchCommand, hp::Command>(m, "DispatchCommand",
            "Command for executing a program using the given group size")
        .def_rw("groupCountX", &hp::DispatchCommand::groupCountX,
            "Amount of groups in X dimension")
        .def_rw("groupCountY", &hp::DispatchCommand::groupCountY,
            "Amount of groups in Y dimension")
        .def_rw("groupCountZ", &hp::DispatchCommand::groupCountZ,
            "Amount of groups in Z dimension");
    
    nb::class_<hp::DispatchIndirectCommand, hp::Command>(m, "DispatchIndirectCommand",
            "Command for executing a program using the group size read from "
            "the provided tensor at given offset")
        .def_rw("tensor", &hp::DispatchIndirectCommand::tensor,
            "Tensor from which to read the group size")
        .def_rw("offset", &hp::DispatchIndirectCommand::offset,
            "Offset into the Tensor in bytes on where to start reading");
    
    nb::class_<hp::Program>(m, "Program",
            "Encapsulates a shader program enabling introspection into its "
            "bindings as well as keeping track of the parameters currently bound "
            "to them. Execution happens trough commands.")
        .def("__init__",
            [](hp::Program* p, nb::bytes code) {
                nb::gil_scoped_release release;
                new (p) hp::Program(getCurrentContext(),
                    std::span<const uint32_t>{
                        reinterpret_cast<const uint32_t*>(code.c_str()),
                        code.size() / 4
                    }
                );
            }, "code"_a,
            "Creates a new program using the shader's byte code"
            "\n\nParameters\n----------\n"
            "code: bytes\n"
            "    Byte code of the program\n")
        .def("__init__",
            [](hp::Program* p, nb::bytes code, nb::bytes spec) {
                nb::gil_scoped_release release;
                new (p) hp::Program(
                    getCurrentContext(),
                    std::span<const uint32_t>{
                        reinterpret_cast<const uint32_t*>(code.c_str()),
                        code.size() / 4
                    },
                    std::span<const std::byte>{
                        reinterpret_cast<const std::byte*>(spec.c_str()),
                        spec.size()
                    }
                );
            }, "code"_a, "specialization"_a,
            "Creates a new program using the shader's byte code"
            "\n\nParameters\n----------\n"
            "code: bytes\n"
            "    Byte code of the program\n"
            "specialization: bytes\n"
            "    Data used for filling in specialization constants")
        .def_prop_ro("localSize",
            [](const hp::Program& p) { return p.getLocalSize(); },
            "Returns the size of the local work group.")
        .def_prop_ro("bindings",
            [](const hp::Program& p){ return p.listBindings(); },
            "Returns a list of all bindings.")
        .def("isBindingBound", [](const hp::Program& p, uint32_t i){
                return p.isBindingBound(i);
            }, "i"_a, "Checks wether the i-th binding is bound")
        .def("isBindingBound", [](const hp::Program& p, std::string_view name) {
                return p.isBindingBound(name);
            }, "name"_a, "Checks wether the binding specified by its name is bound")
        .def("dispatch",
            [](const hp::Program& p, uint32_t x, uint32_t y, uint32_t z)
                -> hp::DispatchCommand
                { return p.dispatch(x, y, z); },
            "x"_a = 1, "y"_a = 1, "z"_a = 1,
            "Dispatches a program execution with the given amount of workgroups."
            "\n\nParameters\n----------\n"
            "x: int, default=1\n"
            "    Number of groups to dispatch in X dimension\n"
            "y: int, default=1\n"
            "    Number of groups to dispatch in Y dimension\n"
            "z: int, default=1\n"
            "    Number of groups to dispatch in Z dimension\n")
        .def("dispatchPush",
            [](const hp::Program& p, nb::bytes push, uint32_t x, uint32_t y, uint32_t z)
                -> hp::DispatchCommand
                {
                    return p.dispatch(
                        std::span<const std::byte>{
                            reinterpret_cast<const std::byte*>(push.c_str()),
                            push.size()
                        },
                        x, y, z
                    );
                }, nb::keep_alive<0,2>(), //keep push bytes as long alive as the dispatch command
            "push"_a, "x"_a = 1, "y"_a = 1, "z"_a = 1,
            "Dispatches a program execution with the given push data and amount of workgroups."
            "\n\nParameters\n----------\n"
            "push: bytes\n"
            "   Data pushed to the dispatch as bytes\n"
            "x: int, default=1\n"
            "    Number of groups to dispatch in X dimension\n"
            "y: int, default=1\n"
            "    Number of groups to dispatch in Y dimension\n"
            "z: int, default=1\n"
            "    Number of groups to dispatch in Z dimension\n")
        .def("dispatchIndirect",
            [](const hp::Program& p, const hp::Tensor<std::byte>& tensor, uint64_t offset)
                -> hp::DispatchIndirectCommand
                {
                    return p.dispatchIndirect(tensor, offset);
                },
            "tensor"_a, "offset"_a = 0,
            "Dispatches a program execution using the amount of workgroups stored "
            "in the given tensor at the given offset. Expects the workgroup size "
            "as three consecutive unsigned 32 bit integers."
            "\n\nParameters\n----------\n"
            "tensor: Tensor\n"
            "    Tensor from which to read the amount of workgroups\n"
            "offset: int, default=0\n"
            "    Offset at which to start reading\n")
        .def("dispatchIndirectPush",
            [](const hp::Program& p, nb::bytes push, const hp::Tensor<std::byte>& tensor, uint64_t offset)
                -> hp::DispatchIndirectCommand
                {
                    return p.dispatchIndirect(
                        std::span<const std::byte>{
                            reinterpret_cast<const std::byte*>(push.c_str()),
                            push.size()
                        },
                        tensor, offset
                    );
                }, nb::keep_alive<0,2>(), //keep push bytes as long alive as the dispatch command
            "push"_a, "tensor"_a, "offset"_a = 0,
            "Dispatches a program execution with the given push data using the amount of "
            "workgroups stored in the given tensor at the given offset."
            "\n\nParameters\n----------\n"
            "push: bytes\n"
            "   Data pushed to the dispatch as bytes\n"
            "tensor: Tensor\n"
            "    Tensor from which to read the amount of workgroups\n"
            "offset: int, default=0\n"
            "    Offset at which to start reading\n")
        .def("__repr__", [](const hp::Program& p) {
            std::ostringstream str;
            auto& ls = p.getLocalSize();
            str << "Program (x: " << ls.x << ", y: " << ls.y << ", z: " << ls.z << ")\n";
            for (auto& b : p.listBindings())
                printBinding(str, b);
            return str.str();
        });
    
    nb::class_<hp::FlushMemoryCommand, hp::Command>(m, "FlushMemoryCommand",
            "Command for flushing memory writes")
        .def("__init__",
            [](hp::FlushMemoryCommand* cmd)
                { new (cmd) hp::FlushMemoryCommand(getCurrentContext()); });
    m.def("flushMemory",
        []() -> hp::FlushMemoryCommand
            { return hp::flushMemory(getCurrentContext()); },
        "Returns a command for flushing memory writes.");
}
