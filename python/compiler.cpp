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
    nb::bind_map<hp::Compiler::HeaderMap>(m, "HeaderMap");

    nb::class_<hp::Compiler>(m, "Compiler")
        .def(nb::init<>())
        .def("addIncludeDir", &hp::Compiler::addIncludeDir, "path"_a,
            "Adds a path to the list of include directories that take part in resolving includes.")
        .def("popIncludeDir", &hp::Compiler::popIncludeDir,
            "Removes the last added include dir from the internal list")
        .def("clearIncludeDir", &hp::Compiler::clearIncludeDir,
            "Clears the internal list of include directories")
        .def("compile", [](const hp::Compiler& c, std::string_view code) -> nb::bytes {
            auto result = c.compile(code);
            return nb::bytes((const char*)result.data(), result.size()*4);
        }, "code"_a, "Compiles the given GLSL code and returns the SPIR-V code as bytes")
        .def("compile", [](const hp::Compiler& c, std::string_view code, const hp::Compiler::HeaderMap& headers) -> nb::bytes {
            auto result = c.compile(code, headers);
            return nb::bytes((const char*)result.data(), result.size()*4);
        }, "code"_a, "headers"_a, "Compiles the given GLSL code using the provided header files and returns the SPIR-V code as bytes");
}
