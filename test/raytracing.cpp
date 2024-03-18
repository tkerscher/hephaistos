#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <array>
#include <cstdint>

#include <hephaistos/hephaistos.hpp>
#include <hephaistos/raytracing.hpp>

#include "validation.hpp"

//code
#include "shader/raytracing.h"

using namespace hephaistos;

namespace {

//EXTENSION

auto Extensions = std::to_array({
    createRaytracingExtension()
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
	0.0f, 0.0f, 1.0f, 1.0f
};
const TransformMatrix BottomTransform = {
    1.0f, 0.0f, 0.0f, 0.0f,
	0.0f, 1.0f, 0.0f, 0.0f,
	0.0f, 0.0f, 1.0f, -1.0f
};
const TransformMatrix LeftTransform = {
    0.0f, 0.0f, 1.0f, -1.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    -1.0f, 0.0f, 0.0f, 0.0f
};
const TransformMatrix RightTransform = {
    0.0f, 0.0f, 1.0f, 1.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    -1.0f, 0.0f, 0.0f, 0.0f
};
const TransformMatrix BackTransform = {
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 1.0f,
    0.0f, -1.0f, 0.0f, 0.0f
};
const TransformMatrix FrontTransform = {
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, -1.0f,
    0.0f, -1.0f, 0.0f, 0.0f
};

struct Result{
    int32_t hit;
    int32_t customIdx;
    int32_t instanceIdx;
    float hitX;
    float hitY;
    float hitZ;
};

// EXPECTED RESULTS

const auto customIdx = std::to_array<uint32_t>({ 17,4,21,15,4,31 });
const auto instanceIdx = std::to_array<uint32_t>({ 0,1,2,3,4,5 });
const auto expectedX = std::to_array<float>({ 0.f, 0.f, -1.f, 1.f, 0.f, 0.f });
const auto expectedY = std::to_array<float>({ 0.f, 0.f, 0.f, 0.f, 1.f, -1.f });
const auto expectedZ = std::to_array<float>({ 1.f, -1.f, 0.f, 0.f, 0.f, 0.f });
const auto eps = 1e-5f;

}

TEST_CASE("ray queries are available in shaders", "[raytracing]") {
    //Check if we have the necessary hardware for this check
    if (!isRaytracingSupported(getDevice(getContext())))
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
    Program program(getContext(), raytracing_code);
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
        REQUIRE_THAT(result[i].hitX, Catch::Matchers::WithinAbs(expectedX[i], eps));
        REQUIRE_THAT(result[i].hitY, Catch::Matchers::WithinAbs(expectedY[i], eps));
        REQUIRE_THAT(result[i].hitZ, Catch::Matchers::WithinAbs(expectedZ[i], eps));
    }

    //check validation errors
    REQUIRE(!hasValidationErrorOccurred());
}
