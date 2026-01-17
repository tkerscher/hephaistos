#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <array>
#include <cstdint>

#include <hephaistos/hephaistos.hpp>
#include <hephaistos/raytracing.hpp>

#include "validation.hpp"

//code
#include "shader/raytracing/rayquery.h"
#include "shader/raytracing/tlas.h"
#include "shader/raytracing/callable.h"
#include "shader/raytracing/hit.h"
#include "shader/raytracing/miss.h"
#include "shader/raytracing/raygen.h"

using namespace hephaistos;

namespace {

//UTIL

template<class T>
std::span<const std::byte> as_bytes(const T& t) {
    return std::as_bytes(std::span<const T>{ std::addressof(t), 1});
}

//EXTENSION

constexpr RayTracingFeatures RAY_TRACING_SUPPORT{
    .query = true,
    .pipeline = true,
    .indirectDispatch = true,
    .positionFetch = true,
    //.invocationReorder = true
};

auto Extensions = std::to_array({
    createRayTracingExtension(RAY_TRACING_SUPPORT)
});

ContextHandle getContext() {
    static ContextHandle context = createEmptyContext();
    if (!context)
        context = createContext(Extensions);
    return context;
}

//GEOMETRY DESCRIPTION

const auto triangle_vertices = std::to_array<float>({
    0.f,  1.f,  0.f,
    -1.f, -1.f, 0.f,
    1.f,  -1.f, 0.f
});

const auto square_vertices = std::to_array<float>({
    1.f, 1.f, 0.f,
    -1.f, -1.f, 0.f,
    1.f, -1.f, 0.f,
    -1.f, 1.f, 0.f
});

const auto square_indices = std::to_array<uint32_t>({
    0, 1, 2,
    0, 3, 1
});

//TRANSFORMATIONS

const TransformMatrix TopTransform = {
    1.0f, 0.0f, 0.0f, 0.0f,
	0.0f, 1.0f, 0.0f, 0.0f,
	0.0f, 0.0f, 1.0f, 5.0f
};
const TransformMatrix BottomTransform = {
    1.0f, 0.0f, 0.0f, 0.0f,
	0.0f, 1.0f, 0.0f, 0.0f,
	0.0f, 0.0f, 1.0f, -5.0f
};
const TransformMatrix LeftTransform = {
    0.0f, 0.0f, 1.0f, -5.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    -1.0f, 0.0f, 0.0f, 0.0f
};
const TransformMatrix RightTransform = {
    0.0f, 0.0f, 1.0f, 5.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    -1.0f, 0.0f, 0.0f, 0.0f
};
const TransformMatrix BackTransform = {
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 5.0f,
    0.0f, -1.0f, 0.0f, 0.0f
};
const TransformMatrix FrontTransform = {
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, -0.5f,
    0.0f, -1.0f, 0.0f, 0.0f
};

struct Result{
    int32_t hit;
    int32_t customIdx;
    int32_t instanceIdx;
    float hitT;
    float hitX;
    float hitY;
    float hitZ;
};

// EXPECTED RESULTS

const auto customIdx = std::to_array<uint32_t>({ 17,4,21,15,4,31 });
const auto instanceIdx = std::to_array<uint32_t>({ 0,1,2,3,4,5 });
const auto expectedT = std::to_array<float>({ 5.f, 5.f, 5.f, 5.f, 5.f, .5f });
const auto expectedX = std::to_array<float>({ 0.f, 0.f, -5.f, 5.f, 0.f, 0.f });
const auto expectedY = std::to_array<float>({ 0.f, 0.f, 0.f, 0.f, 5.f, -.5f });
const auto expectedZ = std::to_array<float>({ 5.f, -5.f, 0.f, 0.f, 0.f, 0.f });
const auto eps = 1e-5f;

const int hitData_1 = 3;
const int hitData_2 = 5;
const auto pipelineResult = std::to_array<int32_t>({
    3, 3, 3, -1, 3, -1, -1, -1, -1,
    3, 3, 3, 3, 0, 5, -1, 5, -1,
    3, 3, 3, -1, 5, -1, -1, -1, -1
});

}

TEST_CASE("can query ray tracing support and properties", "[raytracing]") {
    if (!isRayTracingSupported(RAY_TRACING_SUPPORT))
        SKIP("No ray tracing hardware for testing available.");

    //query properties of first device
    auto devices = enumerateDevices();
    auto& device = devices.front();
    auto features = getRayTracingFeatures(device);
    auto props = getRayTracingProperties(device);
    
    auto context = getContext();
    //check enabled features are the same
    auto enabledFeatures = getEnabledRayTracingFeatures(context);
    bool same =
        enabledFeatures.query == RAY_TRACING_SUPPORT.query &&
        enabledFeatures.pipeline == RAY_TRACING_SUPPORT.pipeline &&
        enabledFeatures.indirectDispatch == RAY_TRACING_SUPPORT.indirectDispatch &&
        enabledFeatures.positionFetch == RAY_TRACING_SUPPORT.positionFetch &&
        enabledFeatures.hitObjects == RAY_TRACING_SUPPORT.hitObjects;
    REQUIRE(same);
    //sensible properties?
    auto enabledProps = getCurrentRayTracingProperties(context);
    bool sensible =
        enabledProps.maxGeometryCount > 0 &&
        enabledProps.maxInstanceCount > 0 &&
        enabledProps.maxPrimitiveCount > 0 &&
        enabledProps.maxAccelerationStructures > 0 &&
        enabledProps.maxRayRecursionDepth > 0 &&
        enabledProps.maxRayDispatchCount > 0 &&
        enabledProps.maxShaderRecordSize;
    REQUIRE(sensible);

    REQUIRE(!hasValidationErrorOccurred());
}

TEST_CASE("ray queries are available in shaders", "[raytracing]") {
    //Check if we have the necessary hardware for this check
    if (!isRayTracingSupported(RAY_TRACING_SUPPORT))
        SKIP("No ray tracing hardware for testing available.");
    
    //build geometries
    GeometryStore store(getContext(), std::to_array<Mesh>({
        { //triangle
            .vertices = std::as_bytes(std::span<const float>(triangle_vertices))
        },
        { //square
            .vertices = std::as_bytes(std::span<const float>(square_vertices)),
            .indices = square_indices
        }
    }));
    auto& triangle = store[0];
    auto& square = store[1];
    //create acceleration structure
    GeometryInstance topInstance{
        .blas_address = triangle.blas_address,
        .transform = TopTransform,
        .customIndex = customIdx[0]
    };
    GeometryInstance bottomInstance{
        .blas_address = square.blas_address,
        .transform = BottomTransform,
        .customIndex = customIdx[1]
    };
    GeometryInstance leftInstance{
        .blas_address = triangle.blas_address,
        .transform = LeftTransform,
        .customIndex = customIdx[2]
    };
    GeometryInstance rightInstance{
        .blas_address = square.blas_address,
        .transform = RightTransform,
        .customIndex = customIdx[3]
    };
    GeometryInstance backInstance{
        .blas_address = triangle.blas_address,
        .transform = BackTransform,
        .customIndex = customIdx[4]
    };
    GeometryInstance frontInstance{
        .blas_address = square.blas_address,
        .transform = FrontTransform,
        .customIndex = customIdx[5]
    };
    auto instances = std::to_array<GeometryInstance>({
        topInstance, bottomInstance,
        leftInstance, rightInstance,
        backInstance, frontInstance
    });
    AccelerationStructure tlas(getContext(), instances);

    //create program
    Buffer<Result> buffer(getContext(), 6);
    Tensor<Result> tensor(getContext(), 6);
    Program program(getContext(), rayquery_code);
    program.bindParameterList(tlas, tensor);

    //run program
    Timeline timeline(getContext());
    beginSequence(timeline)
        .And(program.dispatch(6))
        .Then(retrieveTensor(tensor, buffer))
        .Submit();
    
    //check result
    auto result = buffer.getMemory();
    for (auto i = 0u; i < result.size(); ++i) {
        REQUIRE(result[i].hit == 1);
        REQUIRE(result[i].customIdx == customIdx[i]);
        REQUIRE(result[i].instanceIdx == instanceIdx[i]);
        REQUIRE_THAT(result[i].hitT, Catch::Matchers::WithinAbs(expectedT[i], eps));
        REQUIRE_THAT(result[i].hitX, Catch::Matchers::WithinAbs(expectedX[i], eps));
        REQUIRE_THAT(result[i].hitY, Catch::Matchers::WithinAbs(expectedY[i], eps));
        REQUIRE_THAT(result[i].hitZ, Catch::Matchers::WithinAbs(expectedZ[i], eps));
    }

    //check validation errors
    REQUIRE(!hasValidationErrorOccurred());
}

TEST_CASE("Acceleration structure can be updated", "[raytracing]") {
    //Check if we have the necessary hardware for this check
    if (!isRayTracingSupported(RAY_TRACING_SUPPORT))
        SKIP("No ray tracing hardware for testing available.");

    //build geometries
    GeometryStore store(getContext(), std::to_array<Mesh>({
        { //triangle
            .vertices = std::as_bytes(std::span<const float>(triangle_vertices))
        },
        { //square
            .vertices = std::as_bytes(std::span<const float>(square_vertices)),
            .indices = square_indices
        }
    }));
    auto& triangle = store[0];
    auto& square = store[1];
    //create acceleration structure
    GeometryInstance topInstance{
        .blas_address = triangle.blas_address,
        .transform = TopTransform,
        .customIndex = customIdx[0]
    };
    GeometryInstance bottomInstance{
        .blas_address = square.blas_address,
        .transform = BottomTransform,
        .customIndex = customIdx[1]
    };
    GeometryInstance leftInstance{
        .blas_address = triangle.blas_address,
        .transform = LeftTransform,
        .customIndex = customIdx[2]
    };
    GeometryInstance rightInstance{
        .blas_address = square.blas_address,
        .transform = RightTransform,
        .customIndex = customIdx[3]
    };
    GeometryInstance backInstance{
        .blas_address = triangle.blas_address,
        .transform = BackTransform,
        .customIndex = customIdx[4]
    };
    GeometryInstance frontInstance{
        .blas_address = square.blas_address,
        .transform = FrontTransform,
        .customIndex = customIdx[5]
    };
    auto instances = std::to_array<GeometryInstance>({
        topInstance, bottomInstance,
        leftInstance, rightInstance,
        backInstance, frontInstance
    });

    AccelerationStructure tlas(getContext(), 10);
    tlas.update(instances);

    //create program
    Buffer<Result> buffer(getContext(), 6);
    Tensor<Result> tensor(getContext(), 6);
    Program program(getContext(), rayquery_code);
    program.bindParameterList(tlas, tensor);

    //run program
    Timeline timeline(getContext());
    beginSequence(timeline)
        .And(program.dispatch(6))
        .Then(retrieveTensor(tensor, buffer))
        .Submit();

    //check result
    auto result = buffer.getMemory();
    for (auto i = 0u; i < result.size(); ++i) {
        REQUIRE(result[i].hit == 1);
        REQUIRE(result[i].customIdx == customIdx[i]);
        REQUIRE(result[i].instanceIdx == instanceIdx[i]);
        REQUIRE_THAT(result[i].hitT, Catch::Matchers::WithinAbs(expectedT[i], eps));
        REQUIRE_THAT(result[i].hitX, Catch::Matchers::WithinAbs(expectedX[i], eps));
        REQUIRE_THAT(result[i].hitY, Catch::Matchers::WithinAbs(expectedY[i], eps));
        REQUIRE_THAT(result[i].hitZ, Catch::Matchers::WithinAbs(expectedZ[i], eps));
    }

    //check validation errors
    REQUIRE(!hasValidationErrorOccurred());
}

TEST_CASE("Acceleration structures can be defined on the GPU", "[raytracing]") {
    //Check if we have the necessary hardware for this check
    if (!isRayTracingSupported(RAY_TRACING_SUPPORT))
        SKIP("No ray tracing hardware for testing available.");

    //build geometries
    GeometryStore store(getContext(), std::to_array<Mesh>({
        { //triangle
            .vertices = std::as_bytes(std::span<const float>(triangle_vertices))
        },
        { //square
            .vertices = std::as_bytes(std::span<const float>(square_vertices)),
            .indices = square_indices
        }
    }));
    auto& triangle = store[0];
    auto& square = store[1];
    AccelerationStructure tlas(getContext(), 10);

    //create programs
    Buffer<Result> buffer(getContext(), 6);
    Tensor<Result> tensor(getContext(), 6);
    Program progTlas(getContext(), tlas_code);
    Program progTrace(getContext(), rayquery_code);
    progTrace.bindParameterList(tlas, tensor);
    struct Push {
        uint64_t instanceBufferAddress;
        uint64_t triangleBlasAddress;
        uint64_t squareBlasAddress;
    };
    Push push{ tlas.instanceBufferAddress(), triangle.blas_address, square.blas_address };
    //run program
    Timeline timeline(getContext());
    beginSequence(timeline)
        .And(progTlas.dispatch(push, 6))
        .And(buildAccelerationStructure(tlas))
        .And(progTrace.dispatch(6))
        .Then(retrieveTensor(tensor, buffer))
        .Submit();

    //check result
    auto result = buffer.getMemory();
    for (auto i = 0u; i < result.size(); ++i) {
        REQUIRE(result[i].hit == 1);
        REQUIRE(result[i].customIdx == customIdx[i]);
        REQUIRE(result[i].instanceIdx == instanceIdx[i]);
        REQUIRE_THAT(result[i].hitT, Catch::Matchers::WithinAbs(expectedT[i], eps));
        REQUIRE_THAT(result[i].hitX, Catch::Matchers::WithinAbs(expectedX[i], eps));
        REQUIRE_THAT(result[i].hitY, Catch::Matchers::WithinAbs(expectedY[i], eps));
        REQUIRE_THAT(result[i].hitZ, Catch::Matchers::WithinAbs(expectedZ[i], eps));
    }

    //check validation errors
    REQUIRE(!hasValidationErrorOccurred());
}

TEST_CASE("Ray tracing pipelines can trace rays", "[raytracing]") {
    //Check if we have the necessary hardware for this check
    if (!isRayTracingSupported(RAY_TRACING_SUPPORT))
        SKIP("No ray tracing hardware for testing available.");

    //build geometries
    GeometryStore store(getContext(), std::to_array<Mesh>({
        { //triangle
            .vertices = std::as_bytes(std::span<const float>(triangle_vertices))
        },
        { //square
            .vertices = std::as_bytes(std::span<const float>(square_vertices)),
            .indices = square_indices
        }
        }));
    auto& triangle = store[0];
    auto& square = store[1];
    //create acceleration structure
    GeometryInstance topInstance{
        .blas_address = triangle.blas_address,
        .transform = TopTransform,
        .instanceSBTOffset = 1
    };
    GeometryInstance bottomInstance{
        .blas_address = square.blas_address,
        .transform = BottomTransform,
        .instanceSBTOffset = 0
    };
    GeometryInstance leftInstance{
        .blas_address = triangle.blas_address,
        .transform = LeftTransform,
        .instanceSBTOffset = 0
    };
    GeometryInstance rightInstance{
        .blas_address = square.blas_address,
        .transform = RightTransform,
        .instanceSBTOffset = 1
    };
    GeometryInstance backInstance{
        .blas_address = triangle.blas_address,
        .transform = BackTransform,
        .instanceSBTOffset = 1
    };
    GeometryInstance frontInstance{
        .blas_address = square.blas_address,
        .transform = FrontTransform,
        .instanceSBTOffset = 0
    };
    auto instances = std::to_array<GeometryInstance>({
        topInstance, bottomInstance,
        leftInstance, rightInstance,
        backInstance, frontInstance
        });
    AccelerationStructure tlas(getContext(), instances);

    //create ray tracing pipeline
    RayTracingPipeline pipeline(getContext(),
        std::to_array<RayTracingShader>({
            { RayGenerateShader{ raygen_code } },
            { RayMissShader{ miss_code } },
            { RayHitShader{ hit_code } },
            { CallableShader{ callable_code } }
        })
    );
    //create resources
    Buffer<int32_t> buffer(getContext(), 3 * 3 * 3);
    Tensor<int32_t> tensor(getContext(), 3 * 3 * 3);
    pipeline.bindParameterList(tlas, tensor);

    //create sbt
    auto rayGenTable = pipeline.createShaderBindingTable(0);
    auto missTable = pipeline.createShaderBindingTable(
        std::to_array<ShaderBindingTableEntry>({ {}, {1} })
    );
    auto hitTable = pipeline.createShaderBindingTable(
        std::to_array<ShaderBindingTableEntry>({
            { 2, as_bytes(hitData_1) },
            { 2, as_bytes(hitData_2) }
        })
    );

    //trace rays
    beginSequence(getContext())
        .And(pipeline.traceRays({ rayGenTable, missTable, hitTable }, { 3, 3, 3 }))
        .Then(retrieveTensor(tensor, buffer))
        .Submit();

    //check results
    REQUIRE(std::equal(pipelineResult.begin(), pipelineResult.end(), buffer.getMemory().begin()));

    //check validation errors
    REQUIRE(!hasValidationErrorOccurred());
}

TEST_CASE("Ray tracing pipelines can trace rays indirectly", "[raytracing]") {
    //Check if we have the necessary hardware for this check
    if (!isRayTracingSupported(RAY_TRACING_SUPPORT))
        SKIP("No ray tracing hardware for testing available.");

    //build geometries
    GeometryStore store(getContext(), std::to_array<Mesh>({
        { //triangle
            .vertices = std::as_bytes(std::span<const float>(triangle_vertices))
        },
        { //square
            .vertices = std::as_bytes(std::span<const float>(square_vertices)),
            .indices = square_indices
        }
        }));
    auto& triangle = store[0];
    auto& square = store[1];
    //create acceleration structure
    GeometryInstance topInstance{
        .blas_address = triangle.blas_address,
        .transform = TopTransform,
        .instanceSBTOffset = 1
    };
    GeometryInstance bottomInstance{
        .blas_address = square.blas_address,
        .transform = BottomTransform,
        .instanceSBTOffset = 0
    };
    GeometryInstance leftInstance{
        .blas_address = triangle.blas_address,
        .transform = LeftTransform,
        .instanceSBTOffset = 0
    };
    GeometryInstance rightInstance{
        .blas_address = square.blas_address,
        .transform = RightTransform,
        .instanceSBTOffset = 1
    };
    GeometryInstance backInstance{
        .blas_address = triangle.blas_address,
        .transform = BackTransform,
        .instanceSBTOffset = 1
    };
    GeometryInstance frontInstance{
        .blas_address = square.blas_address,
        .transform = FrontTransform,
        .instanceSBTOffset = 0
    };
    auto instances = std::to_array<GeometryInstance>({
        topInstance, bottomInstance,
        leftInstance, rightInstance,
        backInstance, frontInstance
        });
    AccelerationStructure tlas(getContext(), instances);

    //create ray tracing pipeline
    RayTracingPipeline pipeline(getContext(),
        std::to_array<RayTracingShader>({
            { RayGenerateShader{ raygen_code } },
            { RayMissShader{ miss_code } },
            { RayHitShader{ hit_code } },
            { CallableShader{ callable_code } }
            })
    );
    //create resources
    Buffer<int32_t> buffer(getContext(), 3 * 3 * 3);
    Tensor<int32_t> tensor(getContext(), 3 * 3 * 3);
    pipeline.bindParameterList(tlas, tensor);

    //create sbt
    auto rayGenTable = pipeline.createShaderBindingTable(0);
    auto missTable = pipeline.createShaderBindingTable(
        std::to_array<ShaderBindingTableEntry>({ {}, {1} })
    );
    auto hitTable = pipeline.createShaderBindingTable(
        std::to_array<ShaderBindingTableEntry>({
            { 2, as_bytes(hitData_1) },
            { 2, as_bytes(hitData_2) }
            })
    );

    //trace rays
    auto countTensor = Tensor<uint32_t>(getContext(), std::to_array<uint32_t>({ 3, 3, 3 }));
    beginSequence(getContext())
        .And(pipeline.traceRaysIndirect({ rayGenTable, missTable, hitTable }, countTensor))
        .Then(retrieveTensor(tensor, buffer))
        .Submit();

    //check results
    REQUIRE(std::equal(pipelineResult.begin(), pipelineResult.end(), buffer.getMemory().begin()));

    //check validation errors
    REQUIRE(!hasValidationErrorOccurred());
}
