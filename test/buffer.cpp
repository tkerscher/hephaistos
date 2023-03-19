#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>

#include <hephaistos/buffer.hpp>
#include <hephaistos/command.hpp>

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
