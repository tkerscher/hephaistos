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

using VertexArray = nb::ndarray<float, nb::numpy,
    nb::shape<-1, -1>, nb::c_contig, nb::device::cpu>;
using IndexArray = nb::ndarray<uint32_t, nb::numpy,
    nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
using TransformArray = nb::ndarray<float,
    nb::shape<3,4>, nb::c_contig, nb::device::cpu>;
using TransformArrayOut = nb::ndarray<float, nb::numpy,
    nb::shape<3,4>, nb::c_contig, nb::device::cpu>;
size_t TransformShape[2] = { 3, 4 };

//Wrapper around hephaistos::Mesh with extra numpy
//handles to keep the referenced data alive
struct NumpyMesh : public hp::Mesh {
    VertexArray vertexArray;
    IndexArray indexArray;
};

void registerRaytracing(nb::module_& m) {
    nb::bind_vector<std::vector<NumpyMesh>>(m, "MeshVector",
        "List of Mesh");
    nb::bind_vector<std::vector<hp::Geometry>>(m, "GeometryVector",
        "List of Geometry");
    nb::bind_vector<std::vector<hp::GeometryInstance>>(m, "GeometryInstanceVector",
        "List of GeometryInstance");

    m.def("isRaytracingSupported", &isRaytracingSupported,
        "id"_a.none() = nb::none(),
        "Checks wether any or the given device supports ray tracing.");
    m.def("isRaytracingEnabled",
        []() -> bool { return hp::isRaytracingEnabled(getCurrentContext()); },
        "Checks wether ray tracing was enabled. Note that this creates the context.");
    m.def("enableRaytracing",
        [](bool force) { addExtension(hp::createRaytracingExtension(), force); },
        "force"_a = false,
        "Enables ray tracing. (Lazy) context creation fails if not supported. "
        "Set force=True if an existing context should be destroyed.");
    
    nb::class_<NumpyMesh>(m, "Mesh",
            "Representation of a geometric shape consisting of triangles defined "
            "by their vertices, which can optionally be indexed. "
            "Meshes are used to build Geometries.")
        .def(nb::init<>())
        .def_prop_rw("vertices",
            [](const NumpyMesh& mesh) -> VertexArray { return mesh.vertexArray; },
            [](NumpyMesh& mesh, VertexArray vertices) {
                mesh.vertexArray = vertices;
                size_t bytes = sizeof(float) * vertices.size();
                mesh.vertices = { reinterpret_cast<std::byte*>(vertices.data()), bytes };
                mesh.vertexStride = sizeof(float) * vertices.shape(1);
            },
            "Numpy array holding the vertex data. The first three columns "
            "must be the x, y and z positions.")
        .def_prop_rw("indices",
            [](const NumpyMesh& mesh) -> IndexArray { return mesh.indexArray; },
            [](NumpyMesh& mesh, IndexArray indices) {
                mesh.indexArray = indices;
                mesh.indices = { indices.data(), indices.size() };
            },
            "Optional numpy array holding the indices referencing vertices to create triangles.");
    
    nb::class_<hp::Geometry>(m, "Geometry",
            "Underlying structure Acceleration Structures use to trace rays "
            "against constructed from Meshes.")
        .def(nb::init<>())
        .def_rw("blas_address", &hp::Geometry::blas_address,
            "device address of the underlying blas")
        .def_rw("vertices_address", &hp::Geometry::vertices_address,
            "device address of the vertex buffer or zero if it was discarded")
        .def_rw("indices_address", &hp::Geometry::indices_address,
            "device address of the index buffer, or zero if it was discarded or is non existent");
    
    nb::class_<hp::GeometryInstance>(m, "GeometryInstance",
            "Building blocks of Acceleration Structures containing a reference "
            "to a Geometry along a transformation to be applied to the "
            "underlying mesh as well as additional information.")
        .def(nb::init<>())
        .def_prop_rw("blas_address",
            [](const hp::GeometryInstance& i) -> uint64_t { return i.blas_address; },
            [](hp::GeometryInstance& i, uint64_t v) { i.blas_address = v; },
            "device address of the referenced BLAS/geometry")
        .def_prop_rw("transform",
            [](hp::GeometryInstance& i) -> TransformArrayOut {
                return TransformArrayOut(&i.transform, 2, TransformShape);
            },
            [](hp::GeometryInstance& i, TransformArray a) {
                std::memcpy(&i.transform, a.data(), 3 * 4 * sizeof(float));
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
    
    nb::class_<hp::GeometryStore>(m, "GeometryStore",
            "Handles the creation of Geometries using Meshes and their lifetime "
            "\n\nParameters\n----------\n"
            "meshes: Mesh[]\n"
            "    List of meshes used for creating Geometries\n"
            "keepMeshData: bool, default=True\n",
            "    If True, keeps the mesh data on the GPU after building Geometries.")
        .def("__init__",
            [](hp::GeometryStore* gs, std::vector<NumpyMesh> meshes, bool keepMeshData) {
                nb::gil_scoped_release release;
                //we need to transform numpy meshes to hp::Meshes
                //before passing to the constructor
                std::vector<hp::Mesh> plainMeshes(meshes.size());
                for (auto i = 0u; i < meshes.size(); ++i)
                    plainMeshes[i] = meshes[i];
                new (gs) hp::GeometryStore(getCurrentContext(), plainMeshes, keepMeshData);
            }, "meshes"_a, "keepMeshData"_a = true,
            "Creates a geometry store responsible for managing the BLAS/geometries "
            "used to create and run acceleration structures.")
        .def_prop_ro("geometries",
            [](const hp::GeometryStore& gs) -> const std::vector<hp::Geometry>& {
                return gs.geometries();
            }, "Returns the list of stored geometries")
        .def_prop_ro("size",
            [](const hp::GeometryStore& gs) -> size_t { return gs.size(); },
            "Number of geometries stored")
        .def("createInstance",
            [](const hp::GeometryStore& gs, size_t idx) -> hp::GeometryInstance {
                nb::gil_scoped_release release;
                return gs.createInstance(idx);
            }, "idx"_a, "Creates a new instance of the specified geometry",
            nb::rv_policy::reference_internal);
    
    nb::class_<hp::AccelerationStructure>(m, "AccelerationStructure",
            "Acceleration Structure used by programs to trace rays against a scene. "
            "Consists of multiple instances of various Geometries."
            "\n\nParameters\n---------\n"
            "instances: GeometryInstance[]\n"
            "    list of instances the structure consists of")
        .def("__init__", [](hp::AccelerationStructure* as, std::vector<hp::GeometryInstance> instances) {
            nb::gil_scoped_release release;
            new (as) hp::AccelerationStructure(getCurrentContext(), instances);
        }, "instances"_a,
        "Creates an acceleration structure for consumption in shaders from the given geometry instances.")
        .def("bindParameter",
            [](const hp::AccelerationStructure& as, hp::Program& p, uint32_t b) {
                as.bindParameter(p.getBinding(b));
            }, "program"_a, "binding"_a,
            "Binds the acceleration structure to the program at the given binding")
        .def("bindParameter",
            [](const hp::AccelerationStructure& as, hp::Program& p, std::string_view b) {
                as.bindParameter(p.getBinding(b));
            }, "program"_a, "binding"_a,
            "Binds the acceleration structure to the program at the given binding");
}
