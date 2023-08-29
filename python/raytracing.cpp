#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/bind_vector.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string_view.h>

#include <cstring>

#include <hephaistos/program.hpp>
#include <hephaistos/raytracing.hpp>

#include "context.hpp"

namespace hp = hephaistos;
namespace nb = nanobind;
using namespace nb::literals;

namespace {

auto RaytracingExtension = hp::createRaytracingExtension();

bool isRaytracingSupported(std::optional<uint32_t> id) {
    auto& devices = getDevices();
    if (id) {
        if (id >= devices.size())
            throw std::runtime_error("There is no device with the selected id!");
        return hp::isRaytracingSupported(devices[*id]);
    }
    else {
        //check if any device is supported
        for (auto& dev : devices) {
            if (hp::isRaytracingSupported(dev))
                return true;
        }
        return false;
    }
}

}

using VertexArray = nb::ndarray<nb::numpy, float, nb::shape<nb::any, nb::any>, nb::c_contig, nb::device::cpu>;
using IndexArray = nb::ndarray<nb::numpy, uint32_t, nb::shape<nb::any,3>, nb::c_contig, nb::device::cpu>;
using TransformArray = nb::ndarray<float, nb::shape<3,4>, nb::c_contig, nb::device::cpu>;
using TransformArrayOut = nb::ndarray<nb::numpy, float, nb::shape<3,4>, nb::c_contig, nb::device::cpu>;
size_t TransformShape[2] = { 3, 4 };

//Wrapper around hephaistos::Geometry with extra numpy
//handles to keep the referenced data alive
struct NumpyGeometry : public hp::Geometry {
    VertexArray vertexArray;
    IndexArray indexArray;
};

void registerRaytracing(nb::module_& m) {
    nb::bind_vector<std::vector<hp::GeometryInstance>>(m, "GeometryVector");

    m.def("isRaytracingSupported", &isRaytracingSupported, "id"_a.none() = nb::none(),
        "Checks wether any or the given device supports ray tracing.");
    m.def("isRaytracingEnabled", []() -> bool { return hp::isRaytracingEnabled(getCurrentContext()); },
        "Checks wether ray tracing was enabled. Note that this creates the context.");
    m.def("enableRaytracing", [](bool force) { addExtension(hp::createRaytracingExtension(), force); }, "force"_a = false,
        "Enables ray tracing. (Lazy) context creation fails if not supported. Set force=True if an existing context should be destroyed.");
    
    nb::class_<NumpyGeometry>(m, "Geometry")
        .def(nb::init<>())
        .def_prop_rw("vertices",
            [](const NumpyGeometry& g) -> VertexArray { return g.vertexArray; },
            [](NumpyGeometry& g, VertexArray vertices) {
                g.vertexArray = vertices;
                size_t bytes = sizeof(float) * vertices.shape(0) * vertices.shape(1);
                g.vertices = { reinterpret_cast<std::byte*>(vertices.data()), bytes };
                g.vertexStride = sizeof(float) * vertices.stride(0);
                g.vertexCount = vertices.shape(0);
            }, "Numpy array holding the vertex data. The first three columns must be the x, y and z positions.")
        .def_prop_rw("indices",
            [](const NumpyGeometry& g) -> IndexArray { return g.indexArray; },
            [](NumpyGeometry& g, IndexArray indices) {
                g.indexArray = indices;
                g.indices = { indices.data(), indices.size() };
            }, "Optional numpy array holding the indices referencing vertices to create triangles.");
    
    nb::class_<hp::GeometryInstance>(m, "GeometryInstance")
        .def(nb::init<const NumpyGeometry&>())
        .def_prop_rw("geometry",
            [](hp::GeometryInstance& gi) -> const NumpyGeometry& {
                return static_cast<const NumpyGeometry&>(gi.geometry.get());
            },
            [](hp::GeometryInstance& gi, const NumpyGeometry& g) {
                gi.geometry = std::cref(g);
            },
            "The geometry referenced by this instance.")
        .def_prop_rw("transform",
            [](hp::GeometryInstance& gi) -> TransformArrayOut {
                return TransformArrayOut(&gi.transform, 2, TransformShape);
            },
            [](hp::GeometryInstance& gi, TransformArray t) {
                std::memcpy(&gi.transform, t.data(), 3 * 4 * sizeof(float));
            },
            "The transformation applied to the referenced geometry.")
        .def_prop_rw("customIndex",
            [](hp::GeometryInstance& gi) -> uint32_t { return gi.customIndex; },
            [](hp::GeometryInstance& gi, uint32_t idx) { gi.customIndex = idx; },
            "The custom index of this instance available in the shader.")
        .def_prop_rw("mask",
            [](hp::GeometryInstance& gi) -> uint8_t { return gi.mask; },
            [](hp::GeometryInstance& gi, uint8_t m) { gi.mask = m; },
            "Mask of this instance used for masking ray traces.");
    
    nb::class_<hp::AccelerationStructure>(m, "AccelerationStructure")
        .def("__init__", [](hp::AccelerationStructure* as, std::vector<hp::GeometryInstance> instances) {
            new (as) hp::AccelerationStructure(getCurrentContext(), instances);
        }, "instances"_a, "Creates an acceleration structure for consumption in shaders from the given geometry instances.")
        .def("bindParameter", [](const hp::AccelerationStructure& as, hp::Program& p, uint32_t b) { as.bindParameter(p.getBinding(b)); })
        .def("bindParameter", [](const hp::AccelerationStructure& as, hp::Program& p, std::string_view b) { as.bindParameter(p.getBinding(b)); });
}
