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
        .def_ro("buffer16BitAccess", &hp::TypeSupport::buffer16BitAccess)
        .def_ro("uniform16BitAccess", &hp::TypeSupport::uniform16BitAccess)
        .def_ro("pushConstant16BitAccess", &hp::TypeSupport::pushConstant16BitAccess)
        .def_ro("buffer8BitAccess", &hp::TypeSupport::buffer8BitAccess)
        .def_ro("uniform8BitAccess", &hp::TypeSupport::uniform8BitAccess)
        .def_ro("pushConstant8BitAccess", &hp::TypeSupport::pushConstant8BitAccess)
        .def("__repr__", [](const hp::TypeSupport& t) {
            std::ostringstream str;
            str << std::boolalpha;
            str << "Data Types\n";
            str << "----------------------\n";
            str << " float64: " << !!t.float64 << '\n';
            str << " float16: " << !!t.float16 << '\n';
            str << " int64:   " << !!t.int64 << '\n';
            str << " int16:   " << !!t.int16 << '\n';
            str << " int8:    " << !!t.int8 << '\n';
            str << "Buffer Storage Access\n";
            str << "----------------------\n";
            str << " 16 bit:  " << !!t.buffer16BitAccess << '\n';
            str << "  8 bit:  " << !!t.buffer8BitAccess << '\n';
            str << "Uniform Storage Access\n";
            str << "----------------------\n";
            str << " 16 bit:  " << !!t.uniform16BitAccess << '\n';
            str << "  8 bit:  " << !!t.uniform8BitAccess << '\n';
            str << "Push Constant Access\n";
            str << "----------------------\n";
            str << " 16 bit:  " << !!t.pushConstant16BitAccess << '\n';
            str << "  8 bit:  " << !!t.pushConstant8BitAccess;
            return str.str();
        });
    
    m.def("getSupportedTypes", [](std::optional<uint32_t> id) {
        //check if device was provided
        if (id) {
            return hp::getSupportedTypes(getDevice(*id));
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
            .int8 = types.contains("i8"),
            .buffer16BitAccess = types.contains("B16"),
            .uniform16BitAccess = types.contains("U16"),
            .pushConstant16BitAccess = types.contains("P16"),
            .buffer8BitAccess = types.contains("B8"),
            .uniform8BitAccess = types.contains("U8"),
            .pushConstant8BitAccess = types.contains("P8")
        };

        addExtension(hp::createTypeExtension(t), force);
    }, "types"_a, "force"_a = false,
    "Forces the given types, i.e. devices not supported will be considered"
    "not suitable. Set force=True if an existing context should be destroyed");

    nb::class_<hp::FmaSupport>(m, "FmaSupport",
            "List of types for which the device supports OpFmaKHR")
        .def_ro("float16", &hp::FmaSupport::float16)
        .def_ro("float32", &hp::FmaSupport::float32)
        .def_ro("float64", &hp::FmaSupport::float64)
        .def("__repr__", [](const hp::FmaSupport& fma) {
            std::ostringstream str;
            str << std::boolalpha;
            str << "float16: " << !!fma.float16 << '\n';
            str << "float32: " << !!fma.float32 << '\n';
            str << "float64: " << !!fma.float64;
            return str.str();
        });
    
    m.def("getFmaSupport", [](std::optional<uint32_t> id) {
            if (id) {
                return hp::getFmaSupport(getDevice(*id));
            }
            else {
                return hp::getFmaSupport(getCurrentContext());
            }
        }, "id"_a.none() = nb::none(),
        "Queries the types supporting OpFmaKHR");
    
    m.def("requireFmaSupport", [](const nb::set& types, bool force) {
            hp::FmaSupport fma{
                .float16 = types.contains("f16"),
                .float32 = types.contains("f32"),
                .float64 = types.contains("f64")
            };
            addExtension(hp::createFmaExtension(fma), force);
        }, "types"_a, "force"_a = false,
        "Forces support of OpFmaKHR for the given types, i.e. devices not "
        "meeting this requirement will be considered not suitable. Set "
        "force=True if an existing context should be destroyed.");
}
