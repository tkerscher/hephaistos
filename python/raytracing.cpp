#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/bind_vector.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/variant.h>

#include <sstream>

#include <hephaistos/program.hpp>
#include <hephaistos/raytracing.hpp>

#include "context.hpp"
#include "parameter.hpp"

namespace hp = hephaistos;
namespace nb = nanobind;
using namespace nb::literals;

/********************************* EXTENSION **********************************/

namespace {

void initRayTracingFeatures(
    hp::RayTracingFeatures* features,
    bool query,
    bool pipeline,
    bool indirectDispatch,
    bool positionFetch,
    bool hitObjects
) {
    new (features) hp::RayTracingFeatures();
    features->query = query;
    features->pipeline = pipeline;
    features->indirectDispatch = indirectDispatch;
    features->positionFetch = positionFetch;
    features->hitObjects = hitObjects;
}

bool isRayTracingSupported(hp::RayTracingFeatures features, std::optional<uint32_t> id) {
    if (id) {
        return hp::isRayTracingSupported(getDevice(*id), features);
    }
    else {
        return hp::isRayTracingSupported(features);
    }
}

hp::RayTracingProperties getRayTracingProperties(std::optional<uint32_t> id) {
    if (id) {
        return hp::getRayTracingProperties(getDevice(*id));
    }
    else {
        return hp::getCurrentRayTracingProperties(getCurrentContext());
    }
}

template<class T>
T dict_pop(nb::dict& dict, const char* key, T _default) {
    if (dict.contains(key)) {
        T result = nb::cast<T>(dict[key]);
        nb::del(dict[key]);
        return result;
    }
    else {
        return _default;
    }
}

}

void registerRayTracingExtension(nb::module_& m) {
    nb::class_<hp::RayTracingFeatures>(m, "RayTracingFeatures",
            "List of optional ray tracing features")
        .def_rw("query", &hp::RayTracingFeatures::query, "Support for ray queries")
        .def_rw("pipeline", &hp::RayTracingFeatures::pipeline, "Support for ray tracing pipelines")
        .def_rw("indirectDispatch", &hp::RayTracingFeatures::indirectDispatch, "Support for indirect ray tracing dispatch")
        .def_rw("positionFetch", &hp::RayTracingFeatures::positionFetch, "Support for fetching intersection position in shaders")
        .def_rw("hitObjects", &hp::RayTracingFeatures::hitObjects, "Support for hit objects and shader invocation reorder")
        .def("__str__", [](const hp::RayTracingFeatures& f) {
            std::ostringstream str;
            str << std::boolalpha;
            str << "query:            " << f.query << '\n';
            str << "pipeline:         " << f.pipeline << '\n';
            str << "indirectDispatch: " << f.indirectDispatch << '\n';
            str << "positionFetch:    " << f.positionFetch << '\n';
            str << "hitObjects:       " << f.hitObjects;
            return str.str();
        })
        .def("__init__", &initRayTracingFeatures,
            nb::kw_only(),
            "query"_a = false,
            "pipeline"_a = false,
            "indirectDispatch"_a = false,
            "positionFetch"_a = false,
            "hitObjects"_a = false);
    
    nb::class_<hp::RayTracingProperties>(m, "RayTracingProperties",
            "Additional device properties specific to ray tracing")
        .def_rw("maxGeometryCount", &hp::RayTracingProperties::maxGeometryCount)
        .def_rw("maxInstanceCount", &hp::RayTracingProperties::maxInstanceCount)
        .def_rw("maxPrimitiveCount", &hp::RayTracingProperties::maxPrimitiveCount)
        .def_rw("maxAccelerationStructures", &hp::RayTracingProperties::maxAccelerationStructures)
        .def_rw("maxRayRecursionDepth", &hp::RayTracingProperties::maxRayRecursionDepth)
        .def_rw("maxRayDispatchCount", &hp::RayTracingProperties::maxRayDispatchCount)
        .def_rw("maxShaderRecordSize", &hp::RayTracingProperties::maxShaderRecordSize)
        .def_rw("canReorder", &hp::RayTracingProperties::canReorder)
        .def("__str__", [](const hp::RayTracingProperties& p) {
            std::ostringstream str;
            str << std::boolalpha;

            str << "maxGeometryCount:          " << p.maxGeometryCount << '\n';
            str << "maxInstanceCount:          " << p.maxInstanceCount << '\n';
            str << "maxPrimitiveCount:         " << p.maxPrimitiveCount << '\n';
            str << "maxAccelerationStructures: " << p.maxAccelerationStructures << '\n';
            str << "maxRayRecursionDepth:      " << p.maxRayRecursionDepth << '\n';
            str << "maxRayDispatchCount:       " << p.maxRayDispatchCount << '\n';
            str << "maxShaderRecordSize:       " << p.maxShaderRecordSize << '\n';
            str << "canReorder:                " << p.canReorder;

            return str.str();
        });
    
    m.def("getRayTracingFeatures", [](uint32_t id) -> hp::RayTracingFeatures {
        return hp::getRayTracingFeatures(getDevice(id));
    }, "id"_a, "Returns the ray tracing features supported by the device specified by its id");
    m.def("isRayTracingSupported", &isRayTracingSupported,
        "features"_a, "id"_a = nb::none(),
        "Queries the support of the given ray tracing features");
    m.def("getRayTracingProperties", &getRayTracingProperties,
        "id"_a = nb::none(),
        "Queries the ray tracing properties of the device specified by its id "
        "or of the current context if no id is given");
    m.def("getEnabledRayTracingFeatures", []() -> hp::RayTracingFeatures {
            return hp::getEnabledRayTracingFeatures(getCurrentContext());
        }, "Returns the currently enabled ray tracing features");
    m.def("enableRayTracing",
        [](
            nb::object o,
            bool force,
            nb::kwargs kwargs
        ) {
            //either accept a hp::RayTracingFeature object or build it from kwargs
            hp::RayTracingFeatures features{};
            if (nb::isinstance<hp::RayTracingFeatures>(o)) {
                features = nb::cast<hp::RayTracingFeatures>(o);
            }
            else if (o.is_none()) {
                features = {
                    .query = dict_pop(kwargs, "query", false),
                    .pipeline = dict_pop(kwargs, "pipeline", false),
                    .indirectDispatch = dict_pop(kwargs, "indirectDispatch", false),
                    .positionFetch = dict_pop(kwargs, "positionFetch", false),
                    .hitObjects = dict_pop(kwargs, "hitObjects", false)
                };
                if (kwargs.size() != 0)
                    throw std::runtime_error("Unexpected kwargs argument");
            }
            else {
                throw std::runtime_error("Argument has unexpected type");
            }
            addExtension(hp::createRayTracingExtension(features), force);
        },
        "features"_a.none() = nb::none(),
        nb::kw_only(),
        "force"_a = false,
        "options"_a,
        "Enabled the given ray tracing features. Set `force` to `True`, if an "
        "existing context should be destroyed.");
}

/*************************** ACCELERATION STRUCTURE ***************************/

namespace {
    
using VertexArray = nb::ndarray<float, nb::numpy,
    nb::shape< -1, -1 >, nb::c_contig, nb::device::cpu>;
using IndexArray = nb::ndarray<uint32_t, nb::numpy,
    nb::shape< -1, 3 >, nb::c_contig, nb::device::cpu>;
using TransformArray = nb::ndarray<float,
    nb::shape< 3, 4 >, nb::c_contig, nb::device::cpu>;
using TransformArrayOut = nb::ndarray<float, nb::numpy,
    nb::shape< 3, 4 >, nb::c_contig, nb::device::cpu>;
size_t TransformShape[2] = { 3, 4 };

//Wrapper around hephaistos::Mesh with extra numpy
//handles to keep the referenced data alive
struct NumpyMesh : public hp::Mesh {
    VertexArray vertexArray;
    IndexArray indexArray;
};
    
}

void registerRayTracingStructure(nb::module_& m) {
    nb::bind_vector<std::vector<NumpyMesh>>(m,
        "MeshVector", "List of meshes");
    nb::bind_vector<std::vector<hp::Geometry>>(m,
        "GeometryVector", "List of geometries");
    nb::bind_vector<std::vector<hp::GeometryInstance>>(m,
        "GeometryInstanceVector", "List of geometry instances");

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
            "Device address of the referenced BLAS/geometry. "
            "A value of zero marks the instance as inactive.")
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
    
    nb::class_<hp::GeometryStore, hp::Resource>(m, "GeometryStore",
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
    
    auto acc = nb::class_<hp::AccelerationStructure, hp::Resource>(m, "AccelerationStructure",
            "Acceleration Structure used by programs to trace rays against a scene. "
            "Consists of multiple instances of various Geometries.")
        .def("__init__",
            [](
                hp::AccelerationStructure* as,
                std::vector<hp::GeometryInstance> instances,
                bool freeze
            ) {
                nb::gil_scoped_release release;
                new (as) hp::AccelerationStructure(getCurrentContext(), instances);
            }, "instances"_a, nb::kw_only(), "freeze"_a = false,
            "Creates an acceleration structure for consumption in shaders from "
            "the given geometry instances."
            "\n\nParameters\n---------\n"
            "instances: GeometryInstance[]\n"
            "    list of instances the structure consists of")
        .def("__init__", [](hp::AccelerationStructure* as, uint32_t capacity) {
                nb::gil_scoped_release release;
                new (as) hp::AccelerationStructure(getCurrentContext(), capacity);
            }, "capacity"_a,
            "Creates an acceleration structure with given capacity. "
            "Each instance is initialzed as inactive."
            "\n\nParameters\n---------\n"
            "capacity: int\n"
            "    Maximum amount of instances the acceleration structure can hold")
        .def_prop_ro("capacity", &hp::AccelerationStructure::capacity,
            "Number of instances that can fit in the acceleration structure")
        .def_prop_ro("size", &hp::AccelerationStructure::size,  
            "Current amount of instances in the acceleration structure")
        .def_prop_ro("instanceBufferAddress", &hp::AccelerationStructure::instanceBufferAddress,
            "Device buffer address of where the contiguous array of VkAccelerationStructureInstanceKHR "
            "elements used to build the acceleration structure is stored. Will raise an exception if "
            "the acceleration structure is frozen.")
        .def_prop_ro("frozen", &hp::AccelerationStructure::frozen,
            "Whether the acceleration structure is frozen and cannot be altered.")
        .def("freeze", &hp::AccelerationStructure::freeze,
            "Freezes the acceleration structure preventing further alterations.")
        .def("update", [](hp::AccelerationStructure& as, std::vector<hp::GeometryInstance> instances) {
                nb::gil_scoped_release release;
                as.update(instances);
            }, "instances"_a,
            "Updates the acceleration structure using the given instances. Unused capacity "
            "will be filled with inactive instances. Will raise an exception if amount of "
            "instances exceed the acceleration structure's capacity or if it is frozen.");
    registerArgument(acc);
    
    nb::class_<hp::BuildAccelerationStructureCommand, hp::Command>(m, "BuildAccelerationStructureCommand",
            "Command issuing a rebuild of the corresponding acceleration structure")
        .def(nb::init<const hp::AccelerationStructure&, bool>(),
            "accelerationStructure"_a, nb::kw_only(), "unsafe"_a=false);
    m.def("buildAccelerationStructure", &hp::buildAccelerationStructure,
        "accelerationStructure"_a, nb::kw_only(), "unsafe"_a=false,
        "Creates a command issuing the rebuild of a given acceleration structure."
        "\n\nParameters\n----------\n"
        "accelerationStructure: AccelerationStructure\n"
        "    Acceleration structure to be rebuild.\n"
        "unsafe: bool, default=False\n"
        "    Whether to skip memory barriers.");
}

/**************************** RAY TRACING PIPELINE ****************************/

namespace {

hp::RayTracingShader castShader(nb::handle h) {
    //unfortunately no switch/case, so if/else it is...
    if (nb::isinstance<hp::RayGenerateShader>(h)) {
        return { *nb::cast<const hp::RayGenerateShader*>(h) };
    }
    if (nb::isinstance<hp::RayMissShader>(h)) {
        return { *nb::cast<const hp::RayMissShader*>(h) };
    }
    if (nb::isinstance<hp::RayHitShader>(h)) {
        return { *nb::cast<const hp::RayHitShader*>(h) };
    }
    if (nb::isinstance<hp::CallableShader>(h)) {
        return { *nb::cast<const hp::CallableShader*>(h) };
    }

    //cast failed
    throw std::runtime_error("Invalid shader type");
}

hp::ShaderBindingTableEntry castSBTEntry(nb::handle h) {
    if (h.is_none()) {
        return { ~0u };
    }
    if (nb::isinstance<nb::tuple>(h)) {
        auto t = nb::cast<nb::tuple>(h);
        if (t.size() != 2)
            throw std::runtime_error("Expected a tuple of size 2");
        auto hGroupIdx = nb::handle(t[0]);
        auto hRecord = nb::handle(t[1]);
        if (!nb::isinstance<uint32_t>(hGroupIdx) || !nb::isinstance<nb::bytes>(hRecord))
            throw std::runtime_error("Expected a tuple of type (int, bytes)");
        
        auto groupIdx = nb::cast<uint32_t>(hGroupIdx);
        auto record = nb::cast<nb::bytes>(hRecord);
        return {
            groupIdx,
            { static_cast<const std::byte*>(record.data()), record.size() }
        };
    }
    if (nb::isinstance<uint32_t>(h)) {
        auto groupIdx = nb::cast<uint32_t>(h);
        return { groupIdx };
    }

    //cast failed
    throw std::runtime_error("Could not parse shader binding table entry");
}

hp::ShaderCode castShaderCode(nb::object o) {
    if (o.is_none())
        return {};
    if (nb::isinstance<nb::bytes>(o)) {
        auto code = nb::cast<nb::bytes>(o);
        return {
            std::span<const uint32_t>{
                static_cast<const uint32_t*>(code.data()),
                code.size() / 4
            }
        };
    }
    if (nb::isinstance<nb::tuple>(o)) {
        auto t = nb::cast<nb::tuple>(o);
        if (t.size() != 2)
            throw std::runtime_error("Expected a tuple of size 2");
        auto hCode = nb::handle(t[0]);
        auto hName = nb::handle(t[1]);
        if (!nb::isinstance<nb::bytes>(hCode) || !nb::isinstance<nb::str>(hName))
            throw std::runtime_error("Expected a tuple of type (bytes, str)");
        
        auto code = nb::cast<nb::bytes>(hCode);
        auto name = nb::cast<nb::str>(hName);
        return {
            std::span<const uint32_t>{
                static_cast<const uint32_t*>(code.data()),
                code.size() / 4
            },
            name.c_str()
        };
    }

    //cast failed
    throw std::runtime_error("Could not parse shader code object");
}

template<class T>
void initShader(T* shader, nb::bytes code, std::string_view entryPointName) {
    new (shader) T;
    shader->code.code = {
        static_cast<const uint32_t*>(code.data()),
        code.size() / 4
    };
    shader->code.entryName = entryPointName;
}
//explicit instantiation so we can take the address
template void initShader<hp::RayGenerateShader>(
    hp::RayGenerateShader*, nb::bytes, std::string_view);
template void initShader<hp::RayMissShader>(
    hp::RayMissShader*, nb::bytes, std::string_view);
template void initShader<hp::CallableShader>(
    hp::CallableShader*, nb::bytes, std::string_view);

}

void registerRayTracingPipeline(nb::module_& m) {
    nb::class_<hp::RayGenerateShader>(m, "RayGenerateShader",
            "Shader used to generate rays in a ray tracing pipeline")
        .def("__init__", &initShader<hp::RayGenerateShader>,
            "code"_a, "entryPointName"_a = "main",
            nb::keep_alive<1, 2>(), nb::keep_alive<1, 3>());
    nb::class_<hp::RayMissShader>(m, "RayMissShader",
            "Shader called by ray tracing pipeline when a ray misses")
        .def("__init__", &initShader<hp::RayMissShader>,
            "code"_a, "entryPointName"_a = "main",
            nb::keep_alive<1, 2>(), nb::keep_alive<1, 3>());
    nb::class_<hp::CallableShader>(m, "CallableShader",
            "Callable shader available in ray tracing pipelines")
        .def("__init__", &initShader<hp::CallableShader>,
            "code"_a, "entryPointName"_a = "main",
            nb::keep_alive<1, 2>(), nb::keep_alive<1, 3>());
    nb::class_<hp::RayHitShader>(m, "RayHitShader",
            "Shader called by ray tracing pipeline when a ray hits")
        .def("__init__", [](
                hp::RayHitShader* shader,
                nb::object closest,
                nb::object any
            ) {
                new (shader) hp::RayHitShader;
                shader->closest = castShaderCode(closest);
                shader->any = castShaderCode(any);
            }, "closest"_a, "any"_a.none() = nb::none());
    
    nb::class_<hp::ShaderBindingTable, hp::Resource>(m, "ShaderBindingTable",
            "Shader binding table acting as a sort of vtable for ray tracing pipelines. "
            "See the Vulkan specification for a more detailed explanation.");
    
    nb::class_<hp::TraceRaysCommand, hp::Command>(m, "TraceRaysCommand",
            "Command for issuing rays to be traced using a ray tracing pipeline")
        .def_prop_rw("rayCountX",
            [](const hp::TraceRaysCommand& c) -> uint32_t { return c.rayCount.x; },
            [](hp::TraceRaysCommand& c, uint32_t v) { c.rayCount.x = v; },
            "Amount of rays to dispatch in the x dimension")
        .def_prop_rw("rayCountY",
            [](const hp::TraceRaysCommand& c) -> uint32_t { return c.rayCount.y; },
            [](hp::TraceRaysCommand& c, uint32_t v) { c.rayCount.y = v; },
            "Amount of rays to dispatch in the y dimension")
        .def_prop_rw("rayCountZ",
            [](const hp::TraceRaysCommand& c) -> uint32_t { return c.rayCount.z; },
            [](hp::TraceRaysCommand& c, uint32_t v) { c.rayCount.z = v; },
            "Amount of rays to dispatch in the z dimension");
    nb::class_<hp::TraceRaysIndirectCommand, hp::Command>(m, "TraceRaysIndirectCommand",
            "Command for tracing rays where the amount is read from device memory")
        .def_prop_rw("tensor",
            [](const hp::TraceRaysIndirectCommand& c) -> const hp::Tensor<std::byte>& { return c.tensor.get(); },
            [](hp::TraceRaysIndirectCommand& c, const hp::Tensor<std::byte>& t) { c.tensor = std::cref(t); },
            "Tensor from which to read the amount of rays to trace")
        .def_rw("offset", &hp::TraceRaysIndirectCommand::offset,
            "Offset into the tensor from which to start reading");

    auto pipeline = nb::class_<hp::RayTracingPipeline, hp::Resource>(m, "RayTracingPipeline",
            "Ray tracing pipelines hold a collection of smaller shader stages "
            "needed for tracing rays, namely the ray generation, miss, hit and "
            "callable shaders. These can be arranged as needed, e.g. changing "
            "the ray generation shader or using multiple different hit shaders. "
            "These arrangements are stored in shader binding tables, which can "
            "hold additional data for these shaders allowing to parameterize "
            "the referenced shaders.")
        .def("__init__",
            [](
                hp::RayTracingPipeline* pipeline,
                nb::list shaders,
                uint32_t maxRecursionDepth,
                nb::bytes specialization
            ) {
                //collect all shaders
                std::vector<hp::RayTracingShader> shaderList;
                shaderList.reserve(shaders.size());
                for (nb::handle h : shaders)
                    shaderList.push_back(castShader(h));

                //create ray tracing pipeline
                nb::gil_scoped_release release;
                new (pipeline) hp::RayTracingPipeline(
                    getCurrentContext(),
                    shaderList,
                    std::span<const std::byte>{
                        static_cast<const std::byte*>(specialization.data()),
                        specialization.size()
                    },
                    maxRecursionDepth
                );
            }, "shaders"_a, nb::kw_only(),
            "maxRecursionDepth"_a = 1, "specialization"_a = nb::bytes(nullptr, 0),
            "Creates a new ray tracing pipeline using the specified shaders")
        .def("createShaderBindingTable",
            [](
                const hp::RayTracingPipeline& pipeline,
                nb::list entries
            ) -> hp::ShaderBindingTable {
                //collect all entries
                std::vector<hp::ShaderBindingTableEntry> entryList;
                entryList.reserve(entries.size());
                for (nb::handle h : entries)
                    entryList.push_back(castSBTEntry(h));
                
                //create SBT
                nb::gil_scoped_release release;
                return pipeline.createShaderBindingTable(entryList);
            }, "entries"_a,
            "Creates a new shader binding table containing the specified entries. "
            "Each entry is either the index of the referenced shader or a tuple "
            "consisting of the group index and a bytes object containing the "
            "optional shader record data to be accessed in the shader.")
        .def("createShaderBindingTableRange",
            [](
                const hp::RayTracingPipeline& pipeline,
                uint32_t firstGroupIdx,
                uint32_t count
            ) -> hp::ShaderBindingTable {
                return pipeline.createShaderBindingTable(firstGroupIdx, count);
            },
            "firstGroupIdx"_a, "count"_a = 1,
            "Creates a new shader binding table consisting of the `count`shader "
            "groups starting at `firstGroupIdx` using the same order as during "
            "creation of the ray tracing pipeline")
        .def("traceRays",
            [](
                const hp::RayTracingPipeline& pipeline,
                uint32_t x, uint32_t y, uint32_t z,
                const hp::ShaderBindingTable& rayGenShader,
                const hp::ShaderBindingTable& missShaders,
                const hp::ShaderBindingTable& hitShaders,
                const hp::ShaderBindingTable* callableShaders //nullable
            ) -> hp::TraceRaysCommand {
                hp::ShaderBindings bindings{ rayGenShader, missShaders, hitShaders };
                if (callableShaders) bindings.callableShaders = *callableShaders;
                return pipeline.traceRays(bindings, { x, y, z });
            },
            "x"_a = 1, "y"_a = 1, "z"_a = 1,
            nb::kw_only(),
            "rayGenShaders"_a, "missShaders"_a, "hitShaders"_a,
            "callableShaders"_a.none() = nb::none(),
            nb::keep_alive<0, 5>(),
            nb::keep_alive<0, 6>(),
            nb::keep_alive<0, 7>(),
            nb::keep_alive<0, 8>(),
            "Traces the given amount of rays.\n\n"
            "Parameters\n"
            "----------\n"
            "x: int, default=1\n"
            "   Amount of rays to trace in the x dimension\n"
            "y: int, default=1\n"
            "   Amount of rays to trace in the y dimension\n"
            "z: int, default=1\n"
            "   Amount of rays to trace in the z dimension\n"
            "rayGenShaders: ShaderBindingTable\n"
            "   Reference to the shader creating the rays\n"
            "missShaders: ShaderBindingTable\n"
            "   Reference to the shaders called for missed rays\n"
            "hitShaders: ShaderBindingTable\n"
            "   Reference to the shaders called for hit rays\n"
            "callableShaders: ShaderBindingTable | None, default=None\n"
            "   Optional callable shaders")
        .def("traceRaysPush",
            [](
                const hp::RayTracingPipeline& pipeline,
                nb::bytes push,
                uint32_t x, uint32_t y, uint32_t z,
                const hp::ShaderBindingTable& rayGenShader,
                const hp::ShaderBindingTable& missShaders,
                const hp::ShaderBindingTable& hitShaders,
                const hp::ShaderBindingTable* callableShaders //nullable
            ) -> hp::TraceRaysCommand {
                hp::ShaderBindings bindings{ rayGenShader, missShaders, hitShaders };
                if (callableShaders) bindings.callableShaders = *callableShaders;
                return pipeline.traceRays(bindings, { x, y, z }, {
                    static_cast<const std::byte*>(push.data()), push.size()
                });
            },
            "push"_a, "x"_a = 1, "y"_a = 1, "z"_a = 1,
            nb::kw_only(),
            "rayGenShaders"_a, "missShaders"_a, "hitShaders"_a,
            "callableShaders"_a.none() = nb::none(),
            nb::keep_alive<0, 2>(),
            nb::keep_alive<0, 6>(),
            nb::keep_alive<0, 7>(),
            nb::keep_alive<0, 8>(),
            nb::keep_alive<0, 9>(),
            "Traces the given amount of rays.\n\n"
            "Parameters\n"
            "----------\n"
            "push: bytes\n"
            "   Data pushed to the dispatch as bytes\n"
            "x: int, default=1\n"
            "   Amount of rays to trace in the x dimension\n"
            "y: int, default=1\n"
            "   Amount of rays to trace in the y dimension\n"
            "z: int, default=1\n"
            "   Amount of rays to trace in the z dimension\n"
            "rayGenShaders: ShaderBindingTable\n"
            "   Reference to the shader creating the rays\n"
            "missShaders: ShaderBindingTable\n"
            "   Reference to the shaders called for missed rays\n"
            "hitShaders: ShaderBindingTable\n"
            "   Reference to the shaders called for hit rays\n"
            "callableShaders: ShaderBindingTable | None, default=None\n"
            "   Optional callable shaders")
        .def("traceRaysIndirect",
            [](
                const hp::RayTracingPipeline& pipeline,
                const hp::Tensor<std::byte>& tensor,
                uint64_t offset,
                const hp::ShaderBindingTable& rayGenShader,
                const hp::ShaderBindingTable& missShaders,
                const hp::ShaderBindingTable& hitShaders,
                const hp::ShaderBindingTable* callableShaders //nullable
            ) -> hp::TraceRaysIndirectCommand {
                hp::ShaderBindings bindings{ rayGenShader, missShaders, hitShaders };
                if (callableShaders) bindings.callableShaders = *callableShaders;
                return pipeline.traceRaysIndirect(bindings, tensor, offset);
            },
            "tensor"_a, nb::kw_only(), "offset"_a = 0,
            "rayGenShaders"_a, "missShaders"_a, "hitShaders"_a,
            "callableShaders"_a.none() = nb::none(),
            nb::keep_alive<0, 4>(),
            nb::keep_alive<0, 5>(),
            nb::keep_alive<0, 6>(),
            nb::keep_alive<0, 7>(),
            "Traces rays using the amount stored in the given tensor\n\n"
            "Parameters\n"
            "----------\n"
            "tensor: Tensor\n"
            "   Tensor from which to read the amount of rays to trace\n"
            "offset: int, default=0\n"
            "   Offset into the tensor at which to start reading\n"
            "rayGenShaders: ShaderBindingTable\n"
            "   Reference to the shader creating the rays\n"
            "missShaders: ShaderBindingTable\n"
            "   Reference to the shaders called for missed rays\n"
            "hitShaders: ShaderBindingTable\n"
            "   Reference to the shaders called for hit rays\n"
            "callableShaders: ShaderBindingTable | None, default=None\n"
            "   Optional callable shaders")
        .def("traceRaysIndirectPush",
            [](
                const hp::RayTracingPipeline& pipeline,
                nb::bytes push,
                const hp::Tensor<std::byte>& tensor,
                uint64_t offset,
                const hp::ShaderBindingTable& rayGenShader,
                const hp::ShaderBindingTable& missShaders,
                const hp::ShaderBindingTable& hitShaders,
                const hp::ShaderBindingTable* callableShaders //nullable
            ) -> hp::TraceRaysIndirectCommand {
                hp::ShaderBindings bindings{ rayGenShader, missShaders, hitShaders };
                if (callableShaders) bindings.callableShaders = *callableShaders;
                return pipeline.traceRaysIndirect(bindings, {
                    static_cast<const std::byte*>(push.data()), push.size()
                }, tensor, offset);
            },
            "push"_a, "tensor"_a, nb::kw_only(), "offset"_a = 0,
            "rayGenShaders"_a, "missShaders"_a, "hitShaders"_a,
            "callableShaders"_a.none() = nb::none(),
            nb::keep_alive<0, 2>(),
            nb::keep_alive<0, 5>(),
            nb::keep_alive<0, 6>(),
            nb::keep_alive<0, 7>(),
            nb::keep_alive<0, 8>(),
            "Traces rays using the amount stored in the given tensor\n\n"
            "Parameters\n"
            "----------\n"
            "push: bytes\n"
            "   Data pushed to the dispatch as bytes\n"
            "tensor: Tensor\n"
            "   Tensor from which to read the amount of rays to trace\n"
            "offset: int, default=0\n"
            "   Offset into the tensor at which to start reading\n"
            "rayGenShaders: ShaderBindingTable\n"
            "   Reference to the shader creating the rays\n"
            "missShaders: ShaderBindingTable\n"
            "   Reference to the shaders called for missed rays\n"
            "hitShaders: ShaderBindingTable\n"
            "   Reference to the shaders called for hit rays\n"
            "callableShaders: ShaderBindingTable | None, default=None\n"
            "   Optional callable shaders")
        .def("__str__", [](const hp::RayTracingPipeline& pipeline) {
            std::ostringstream str;
            str << "Ray Tracing Pipeline";
            for (auto& b : pipeline.listBindings()) {
                str << '\n';
                detail::printBinding(str, b);
            }
            return str.str();
        });
    registerBindingTarget(pipeline);
}

void registerRayTracing(nb::module_& m) {
    registerRayTracingExtension(m);
    registerRayTracingStructure(m);
    registerRayTracingPipeline(m);
}
