#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include <format>
#include <sstream>

#include <hephaistos/program.hpp>
#include <hephaistos/raytracing/pipeline.hpp>

namespace hp = hephaistos;
namespace nb = nanobind;
using namespace nb::literals;

namespace detail {

inline void printBindingType(std::ostringstream& str, hp::ParameterType type) {
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

inline void printBinding(std::ostringstream& str, const hp::BindingTraits& b) {
    str << b.binding << ": " << b.name << " (";
    printBindingType(str, b.type);
    if (b.count > 1)
        str << '[' << b.count << ']';
    str << ")";
}
    
}

//Nanobind does not support multiple inheritance
//That's why we simply duplicate the bindings for each 

template<class T>
void registerArgument(nb::class_<T, hp::Resource>& c) {
    //program
    c.def("bindParameter",
        [](const T& t, hp::Program& p, uint32_t b) {
            p.bindParameter(t, b);
        }, "program"_a, "binding"_a,
        "Binds this at the given binding as parameter");
    c.def("bindParameter",
        [](const T& t, hp::Program& p, std::string_view b) {
            p.bindParameter(t, b);
        }, "program"_a, "binding"_a,
        "Binds this at the given binding as parameter");
    //ray tracing pipeline
    c.def("bindParameter",
        [](const T& t, hp::RayTracingPipeline& p, uint32_t b) {
            p.bindParameter(t, b);
        }, "pipeline"_a, "binding"_a,
        "Binds this at the given binding as parameter");
    c.def("bindParameter",
        [](const T& t, hp::RayTracingPipeline& p, std::string_view b) {
            p.bindParameter(t, b);
        }, "pipeline"_a, "binding"_a,
        "Binds this at the given binding as parameter");
}

template<class T>
void registerBindingTarget(nb::class_<T, hp::Resource>& c) {
    c.def_prop_ro("bindings",
        [](const T& t) { return t.listBindings(); },
        "List of all bindings");
    c.def("hasBinding",
        [](const T& t, std::string_view name) { return t.hasBinding(name); },
        "name"_a, "Checks whether a binding with the given name exists");
    c.def("isBindingBound",
        [](const T& t, uint32_t i) { return t.isBindingBound(i); },
        "i"_a, "Checks whether the i-th binding is bound");
    c.def("isBindingBound",
        [](const T& t, std::string_view name) { return t.isBindingBound(name); },
        "name"_a, "Checks whether the binding specified by its name is bound");
    c.def("bindParams",
        [](T& t, nb::args params, nb::kwargs namedparams) {
            auto self = nb::find(t);
            for (auto i = 0; i < params.size(); ++i)
                params[i].attr("bindParameter")(self, i);
            for (auto kv : namedparams) {
                if (kv.first.is_none())
                    continue;
                if (t.hasBinding(nb::str(kv.first).c_str())) {
                    if (nb::hasattr(kv.second, "bindParameter"))
                        kv.second.attr("bindParameter")(self, kv.first);
                    else
                        throw nb::attribute_error(std::format(
                            "Value passed to binding \"{}\" has no attribute \"bindParameter\"",
                            nb::str(kv.first).c_str()
                        ).c_str());
                }
            }
        },
        "Binds the given parameters based on their index if passed as positional "
        "argument or based on their name if passed as keyword argument. Named "
        "parameters without a corresponding binding in the program are ignored.");
}
