#include <nanobind/nanobind.h>

#include <hephaistos/bindings.hpp>

#include "parameter.hpp"

namespace hp = hephaistos;
namespace nb = nanobind;
using namespace nb::literals;

void registerBindingModule(nb::module_& m) {
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
            detail::printBinding(str, b);
            return str.str();
        });
}
