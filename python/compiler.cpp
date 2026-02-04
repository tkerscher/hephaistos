#include <nanobind/nanobind.h>
#include <nanobind/stl/bind_map.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>

#include <hephaistos/compiler.hpp>

namespace hp = hephaistos;
namespace nb = nanobind;
using namespace nb::literals;

void registerCompilerModule(nb::module_& m) {
    nb::enum_<hp::ShaderStage>(m, "ShaderStage")
        .value("COMPUTE", hp::ShaderStage::COMPUTE)
        .value("RAYGEN", hp::ShaderStage::RAYGEN)
        .value("INTERSECT", hp::ShaderStage::INTERSECT)
        .value("ANY_HIT", hp::ShaderStage::ANY_HIT)
        .value("CLOSEST_HIT", hp::ShaderStage::CLOSEST_HIT)
        .value("MISS", hp::ShaderStage::MISS)
        .value("CALLABLE", hp::ShaderStage::CALLABLE);

    nb::bind_map<hp::HeaderMap>(m, "HeaderMap",
        "Dict mapping filepaths to shader source code. "
        "Consumed by Compiler to resolve include directives.");

    nb::class_<hp::Compiler>(m, "Compiler",
            "Compiler for generating SPIR-V byte code used by Programs from "
            "shader code written in GLSL. Has additional methods to handle "
            "includes.")
        .def(nb::init<>())
        .def("addIncludeDir", &hp::Compiler::addIncludeDir, "path"_a,
            "Adds a path to the list of include directories "
            "that take part in resolving includes.")
        .def("popIncludeDir", &hp::Compiler::popIncludeDir,
            "Removes the last added include dir from the internal list")
        .def("clearIncludeDir", &hp::Compiler::clearIncludeDir,
            "Clears the internal list of include directories")
        .def("createSession",
            [](const hp::Compiler& c) -> hp::CompilerSession {
                return hp::CompilerSession(c);
            }, "Creates a new compiler session")
        .def("compile",
            [](
                const hp::Compiler& c,
                std::string_view code,
                hp::ShaderStage stage
            ) -> nb::bytes {
                std::vector<uint32_t> result;
                {
                    //release gil during compilation
                    nb::gil_scoped_release release;
                    result = c.compile(code, stage);
                }
                return nb::bytes((const char*)result.data(), result.size()*4);
            }, "code"_a, nb::kw_only(), "stage"_a = hp::ShaderStage::COMPUTE,
            "Compiles the given GLSL code and returns the SPIR-V code as bytes")
        .def("compile", [](
                    const hp::Compiler& c,
                    std::string_view code,
                    const hp::HeaderMap& headers,
                    hp::ShaderStage stage
                ) -> nb::bytes {
                    std::vector<uint32_t> result;
                    {
                        //release GIL during compilation
                        nb::gil_scoped_release release;
                        result = c.compile(code, headers, stage);
                    }
                    return nb::bytes((const char*)result.data(), result.size()*4);
                },
            "code"_a, "headers"_a, nb::kw_only(), "stage"_a = hp::ShaderStage::COMPUTE,
            "Compiles the given GLSL code using the provided header files "
            "and returns the SPIR-V code as bytes");
    
    nb::class_<hp::CompilerSession>(m, "CompilerSession",
            "Ensures code compiled within a single session share the same pipeline layout. "
            "That is, bindings with the same name will be assigned the same binding point "
            "across all compilations.")
        .def("compile",
            [](
                hp::CompilerSession& c,
                std::string_view code,
                hp::ShaderStage stage
            ) -> nb::bytes {
                std::vector<uint32_t> result;
                {
                    //release gil during compilation
                    nb::gil_scoped_release release;
                    result = c.compile(code, stage);
                }
                return nb::bytes((const char*)result.data(), result.size()*4);
            }, "code"_a, nb::kw_only(), "stage"_a = hp::ShaderStage::COMPUTE,
            "Compiles the given GLSL code and returns the SPIR-V code as bytes")
        .def("compile", [](
                    hp::CompilerSession& c,
                    std::string_view code,
                    const hp::HeaderMap& headers,
                    hp::ShaderStage stage
                ) -> nb::bytes {
                    std::vector<uint32_t> result;
                    {
                        //release GIL during compilation
                        nb::gil_scoped_release release;
                        result = c.compile(code, headers, stage);
                    }
                    return nb::bytes((const char*)result.data(), result.size()*4);
                },
            "code"_a, "headers"_a, nb::kw_only(), "stage"_a = hp::ShaderStage::COMPUTE,
            "Compiles the given GLSL code using the provided header files "
            "and returns the SPIR-V code as bytes");
}
