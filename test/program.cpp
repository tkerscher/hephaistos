#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>

#include <hephaistos/buffer.hpp>
#include <hephaistos/command.hpp>
#include <hephaistos/context.hpp>
#include <hephaistos/image.hpp>
#include <hephaistos/program.hpp>

//shader code
#include "shader/image.h"
#include "shader/push.h"
#include "shader/sbo.h"
#include "shader/spec.h"
#include "shader/ubo.h"

using namespace hephaistos;

namespace {

ContextHandle getContext() {
    static ContextHandle context = createEmptyContext();
    if (!context)
        context = createContext();
    return context;
}

struct DataStruct{
    int32_t const0 = 15;
    int32_t const1 = 4;
    int32_t const2 = 32;
};
DataStruct dataStruct;

std::array<int32_t, 3> data = { { 15, 4, 32 } };

std::array<int32_t, 3> dataIdx = { { 0, 1, 2 } };

}

TEST_CASE("program can use specialization constants", "[program]") {
    Buffer<int32_t> buffer(getContext(), 3);
    Tensor<int32_t> tensor(getContext(), 3);
    Program program(getContext(), spec_code, dataStruct);
    program.bindParameterList(tensor);

    Timeline timeline(getContext());
    beginSequence(timeline)
        .And(program.dispatch(3))
        .Then(retrieveTensor(tensor, buffer))
        .Submit();
    
    REQUIRE(std::equal(data.begin(), data.end(), buffer.getMemory().begin()));

    REQUIRE(!hasValidationErrorOccurred());
}

TEST_CASE("program can use push constants", "[program]") {
    Buffer<int32_t> buffer(getContext(), 3);
    Tensor<int32_t> tensor(getContext(), 3);
    Program program(getContext(), push_code);
    program.bindParameterList(tensor);

    Timeline timeline(getContext());
    beginSequence(timeline)
        .And(program.dispatch(dataStruct, 3))
        .Then(retrieveTensor(tensor, buffer))
        .Submit();
    
    REQUIRE(std::equal(data.begin(), data.end(), buffer.getMemory().begin()));

    REQUIRE(!hasValidationErrorOccurred());
}

TEST_CASE("program can use uniform buffers", "[program]") {
    Buffer<int32_t> buffer(getContext(), 3);
    Tensor<int32_t> tensor(getContext(), 3);
    Tensor<DataStruct> ubo(getContext(), dataStruct);
    Program program(getContext(), ubo_code);
    program.bindParameterList(ubo, tensor);

    Timeline timeline(getContext());
    beginSequence(timeline)
        .And(program.dispatch(3))
        .Then(retrieveTensor(tensor, buffer))
        .Submit();
    
    REQUIRE(std::equal(data.begin(), data.end(), buffer.getMemory().begin()));

    REQUIRE(!hasValidationErrorOccurred());
}

TEST_CASE("program can use storage buffers", "[program]") {
    Buffer<int32_t> buffer(getContext(), 3);
    Tensor<int32_t> tensor(getContext(), 3);
    Program program(getContext(), sbo_code);
    program.bindParameterList(tensor);

    Timeline timeline(getContext());
    beginSequence(timeline)
        .And(program.dispatch(3))
        .Then(retrieveTensor(tensor, buffer))
        .Submit();
    
    REQUIRE(std::equal(dataIdx.begin(), dataIdx.end(), buffer.getMemory().begin()));

    REQUIRE(!hasValidationErrorOccurred());
}

TEST_CASE("program can use storage image", "[program]") {
    Buffer<int32_t> buffer(getContext(), 3);
    Image image(getContext(), ImageFormat::R32_SINT, 3);
    Program program(getContext(), image_code);
    program.bindParameterList(image);

    Timeline timeline(getContext());
    beginSequence(timeline)
        .And(program.dispatch(3))
        .Then(retrieveImage(image, buffer))
        .Submit();

    REQUIRE(std::equal(dataIdx.begin(), dataIdx.end(), buffer.getMemory().begin()));

    REQUIRE(!hasValidationErrorOccurred());
}
