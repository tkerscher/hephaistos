#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>

#include <hephaistos/buffer.hpp>
#include <hephaistos/command.hpp>
#include <hephaistos/program.hpp>

// shader code
#include "shader/buffer_array.h"

using namespace hephaistos;

namespace {

ContextHandle getContext() {
    static ContextHandle context = createEmptyContext();
    if (!context)
        context = createContext();
    return context;
}

std::array<int, 10> data = { {
    10,-5,6,45,12,122,2'147'483'647, 789, 1500, -45123
} };

}

TEST_CASE("typed buffers handle appropriate memory", "[buffer]") {
    Buffer<int> buffer(getContext(), 10);

    SECTION("size queries are correct") {
        REQUIRE(buffer.size_bytes() == (sizeof(int) * 10));
        REQUIRE(buffer.size() == 10);
    }
    
    SECTION("memory span has correct size") {
        auto memory = buffer.getMemory();

        REQUIRE(memory.size_bytes() == (sizeof(int) * 10));
        REQUIRE(memory.size() == 10);
    }

    SECTION("buffer is writable") {
        std::copy(data.begin(), data.end(), buffer.getMemory().begin());
        REQUIRE(std::equal(data.begin(), data.end(), buffer.getMemory().begin()));
    }

    REQUIRE(!hasValidationErrorOccurred());
}

TEST_CASE("buffer can be initialized with data", "[buffer]") {
    Buffer buffer(getContext(), data);

    REQUIRE(buffer.size() == data.size());
    REQUIRE(buffer.size_bytes() == (sizeof(int) * data.size()));
    REQUIRE(std::equal(data.begin(), data.end(), buffer.getMemory().begin()));

    REQUIRE(!hasValidationErrorOccurred());
}

TEST_CASE("tensor know about their size", "[buffer]") {
    Tensor<int> tensor(getContext(), 10);

    REQUIRE(tensor.size() == 10);
    REQUIRE(tensor.size_bytes() == (sizeof(int) * 10));

    REQUIRE(!hasValidationErrorOccurred());
}

TEST_CASE("buffers and tensors can be copied into each other", "[buffer]") {
    Buffer<int> bufferIn(getContext(), 10);
    Buffer<int> bufferOut(getContext(), 10);
    Tensor<int> tensor(getContext(), 10);

    std::copy(data.begin(), data.end(), bufferIn.getMemory().data());
    Timeline timeline(getContext());
    beginSequence(timeline)
        .And(updateTensor(bufferIn, tensor))
        .Then(retrieveTensor(tensor, bufferOut))
        .Submit();
    
    REQUIRE(std::equal(data.begin(), data.end(), bufferOut.getMemory().begin()));

    REQUIRE(!hasValidationErrorOccurred());
}

TEST_CASE("tensors can be initialized with data", "[buffer]") {
    Tensor<int> tensor(getContext(), data);
    Buffer<int> buffer(getContext(), tensor.size());

    //check sizes
    REQUIRE(tensor.size() == data.size());
    REQUIRE(buffer.size() == data.size());
    REQUIRE(tensor.size_bytes() == (sizeof(int) * data.size()));

    //copy back data
    Timeline timeline(getContext());
    beginSequence(timeline)
        .And(retrieveTensor(tensor, buffer))
        .Submit();
    
    //compare data
    REQUIRE(std::equal(data.begin(), data.end(), buffer.getMemory().begin()));

    REQUIRE(!hasValidationErrorOccurred());
}

TEST_CASE("tensors can be bundled into arrays", "[buffer]") {
    Tensor<int> t1(getContext(), std::to_array({ 1, 2, 3, 4 }));  //10
    Tensor<int> t2(getContext(), std::to_array({ 17, 5, 8, 4 })); //34
    Tensor<int> t3(getContext(), std::to_array({ 1, 12, 0, 9 })); //22

    SECTION("arrays can be created with an initializer list") {
        ArgumentArray<Tensor<std::byte>> arr{ t1, t2, t3 };

        REQUIRE(arr.size() == 3);
    }

    SECTION("arrays can be pushed into") {
        ArgumentArray<Tensor<std::byte>> arr;

        REQUIRE(arr.size() == 0);
        REQUIRE(arr.empty());

        arr.push_back(t1);

        REQUIRE(arr.size() == 1);
        REQUIRE(!arr.empty());

        arr.push_back(t1);
        arr.push_back(t2);
        arr.push_back(t1);

        REQUIRE(arr.size() == 4);
    }

    SECTION("arrays are correctly bound to shaders") {
        ArgumentArray<Tensor<std::byte>> arr{ t1, t2, t1, t3 };
        Tensor<int> res(getContext(), 4);
        Buffer<int> buf(getContext(), 4);

        Program program(getContext(), buffer_array_code);
        program.bindParameterList(res, arr);

        beginSequence(getContext())
            .And(program.dispatch(4))
            .Then(retrieveTensor(res, buf))
            .Submit();

        auto p = buf.getMemory();

        REQUIRE(p[0] == 10);
        REQUIRE(p[1] == 34);
        REQUIRE(p[2] == 10);
        REQUIRE(p[3] == 22);
    }

    REQUIRE(!hasValidationErrorOccurred());
}
