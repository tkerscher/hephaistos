#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <tuple>

#include <hephaistos/buffer.hpp>
#include <hephaistos/command.hpp>

#include "validation.hpp"

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

TEST_CASE("buffers can be moved and destroyed", "[buffer}") {
    Buffer<int> buffer(getContext(), 10);
    Buffer<int> buffer2(std::move(buffer));

    REQUIRE(!buffer);
    REQUIRE(buffer.size() == 0);
    REQUIRE(buffer.size_bytes() == 0);
    REQUIRE(!buffer.getContext());
    REQUIRE(buffer2);
    REQUIRE(buffer2.size() == 10);
    REQUIRE(buffer2.size_bytes() == 40);
    REQUIRE(buffer2.getContext());

    buffer = std::move(buffer2);

    REQUIRE(buffer);
    REQUIRE(buffer.size() == 10);
    REQUIRE(buffer.size_bytes() == 40);
    REQUIRE(buffer.getContext());
    REQUIRE(!buffer2);
    REQUIRE(buffer2.size() == 0);
    REQUIRE(buffer2.size_bytes() == 0);
    REQUIRE(!buffer2.getContext());
    
    buffer.destroy();

    REQUIRE(!buffer);
    REQUIRE(buffer.size() == 0);
    REQUIRE(buffer.size_bytes() == 0);
    REQUIRE(!buffer.getContext());

    REQUIRE(!hasValidationErrorOccurred());
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

TEST_CASE("tensors can be mapped", "[buffer]") {
    Tensor<int> tensor(getContext(), 10, true);
    if (!tensor.isMapped()) {
        SKIP("device has no host visible, devive local memory");
    }
    Buffer<int> buffer(getContext(), 10);
    auto mem = buffer.getMemory();

    std::memset(mem.data(), 0, 40);
    std::memcpy(tensor.getMemory().data(), data.data(), 40);

    execute(getContext(), retrieveTensor(tensor, buffer));

    REQUIRE(std::equal(data.begin(), data.end(), mem.begin()));

    REQUIRE(!hasValidationErrorOccurred());
}

TEST_CASE("mapped tensors can be copied to and from", "[buffer]") {
    std::array<int, 10> dst;

    Tensor<int> tensor(getContext(), 10, true);

    //We dont know at what hardware we're running
    //and what decision the allocator will make
    //but let's call this function anyway to see
    //if it at least does not crash
    std::ignore = tensor.isNonCoherent();

    tensor.update(data);
    tensor.flush(); //superficial, but let's call it somewhere

    tensor.invalidate(); //superficial, but let's call it somewhere
    tensor.retrieve(dst);
       
    REQUIRE(std::equal(data.begin(), data.end(), dst.begin()));

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

TEST_CASE("unsafe copies are supported", "[buffer]") {
    //we cant test undefined behaviour. However, by putting each copy command
    //in separate steps, i.e. different submission, one can ensure updating
    //finished before retrieve starts

    Buffer<int> bufferIn(getContext(), 10);
    Buffer<int> bufferOut(getContext(), 10);
    Tensor<int> tensor(getContext(), 10);

    std::copy(data.begin(), data.end(), bufferIn.getMemory().data());
    Timeline timeline(getContext());
    beginSequence(timeline)
        .And(updateTensor(bufferIn, tensor, { .unsafe = true }))
        .Then(retrieveTensor(tensor, bufferOut, { .unsafe = true }))
        .Submit();

    REQUIRE(std::equal(data.begin(), data.end(), bufferOut.getMemory().begin()));

    REQUIRE(!hasValidationErrorOccurred());
}

TEST_CASE("subregions of buffer and tensors can be copied into each other", "[buffer]") {
    Buffer<int> bufferIn(getContext(), 10);
    Buffer<int> bufferOut(getContext(), 10);
    Tensor<int> tensor(getContext(), 10);

    std::copy(data.begin(), data.end(), bufferIn.getMemory().data());
    std::memset(bufferOut.getMemory().data(), 0, 40);

    beginSequence(getContext())
        .And(updateTensor(bufferIn, tensor, { .bufferOffset = 20, .size = 20 }))
        .And(updateTensor(bufferIn, tensor, { .tensorOffset = 20, .size = 20 }))
        .Then(retrieveTensor(tensor, bufferOut, { .bufferOffset = 8, .tensorOffset = 12, .size = 24 }))
        .Submit();

    auto scrambled = std::to_array({
        0, 0, 1500, -45123, 10, -5, 6, 45, 0, 0 
    });
    REQUIRE(std::equal(scrambled.begin(), scrambled.end(), bufferOut.getMemory().begin()));

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

TEST_CASE("tensors have a device address", "[buffer]") {
    Tensor<int> tensor(getContext(), 16);

    REQUIRE(tensor.address() != 0);

    //TODO: Should we test the address via a short shader?

    REQUIRE(!hasValidationErrorOccurred());
}

TEST_CASE("tensors can be filled with constant data", "[buffer]") {
    Tensor<int> tensor(getContext(), 16);
    Buffer<int> buffer(getContext(), 16);
    auto mem = buffer.getMemory();

    beginSequence(getContext())
        .And(clearTensor(tensor, { .data = 5 }))
        .Then(retrieveTensor(tensor, buffer))
        .Submit();

    REQUIRE(std::all_of(mem.begin(), mem.end(), [](int i) -> bool { return i == 5; }));

    beginSequence(getContext())
        .And(clearTensor(tensor, { .offset = 32, .size = 16, .data = 12 }))
        .Then(retrieveTensor(tensor, buffer))
        .Submit();

    std::array<int, 16> data{ {
        5, 5, 5, 5,
        5, 5, 5, 5,
        12, 12, 12, 12,
        5, 5, 5, 5
    } };
    REQUIRE(std::equal(data.begin(), data.end(), mem.begin()));
}
