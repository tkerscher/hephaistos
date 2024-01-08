#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>

#include <sstream>

#include <hephaistos/types.hpp>

#include "context.hpp"

namespace hp = hephaistos;
namespace nb = nanobind;
using namespace nb::literals;

void registerTypeModule(nb::module_& m) {
    nb::class_<hp::TypeSupport>(m, "TypeSupport",
            "List of optional type support in programs")
        .def_ro("float64", &hp::TypeSupport::float64)
        .def_ro("float16", &hp::TypeSupport::float16)
        .def_ro("int64", &hp::TypeSupport::int64)
        .def_ro("int16", &hp::TypeSupport::int16)
        .def_ro("int8", &hp::TypeSupport::int8)
        .def("__repr__", [](const hp::TypeSupport& t) {
            std::ostringstream str;
            str << std::boolalpha;
            str << "float64: " << !!t.float64 << '\n';
            str << "float16: " << !!t.float16 << '\n';
            str << "int64:   " << !!t.int64 << '\n';
            str << "int16:   " << !!t.int16 << '\n';
            str << "int8:    " << !!t.int8;
            return str.str();
        });
    
    m.def("getSupportedTypes", [](std::optional<uint32_t> id) {
        //check if device was provided
        if (id) {
            uint32_t idx = id.value();
            auto& devices = getDevices();
            return hp::getSupportedTypes(devices[idx]);
        }
        else {
            //query with current context
            return hp::getSupportedTypes(getCurrentContext());
        }
    }, "id"_a.none() = nb::none(),
    "Queries the supported extended types");

    m.def("requireTypes", [](const nb::set& types, bool force) {
        hp::TypeSupport t{
            .float64 = types.contains("f64"),
            .float16 = types.contains("f16"),
            .int64 = types.contains("i64"),
            .int16 = types.contains("i16"),
            .int8 = types.contains("i8")
        };

        addExtension(hp::createTypeExtension(t), force);
    }, "types"_a, "force"_a = false,
    "Forces the given types, i.e. devices not supported will be considered"
    "not suitable. Set force=True if an existing context should be destroyed");
}
