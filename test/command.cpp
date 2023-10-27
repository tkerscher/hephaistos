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

}

TEST_CASE("timeline value can be set and queried", "[command]") {
    Timeline timeline(getContext());

    SECTION("Timeline value can be set") {
        timeline.setValue(10);
        REQUIRE(timeline.getValue() == 10);
    }

    SECTION("Timeline value can be waited for") {
        REQUIRE(!timeline.waitValue(10, 100));
        timeline.setValue(10);
        REQUIRE(timeline.waitValue(10, 100));
    }

    REQUIRE(!hasValidationErrorOccurred());
}

TEST_CASE("timeline can start at an arbitrary value", "[command]") {
    Timeline timeline(getContext(), 15);
    REQUIRE(timeline.getValue() == 15);

    REQUIRE(!hasValidationErrorOccurred());
}

//sequences are plenty tested in other places
//here we just test wether we can implement waitings

TEST_CASE("sequences can wait for the cpu", "[command]") {
    Timeline timeline(getContext());

    auto submission = beginSequence(timeline).WaitFor(5).Submit();

    REQUIRE(!submission.wait(100));
    timeline.setValue(2);
    REQUIRE(!submission.wait(100));
    timeline.setValue(5);
    REQUIRE(submission.wait(100));

    REQUIRE(!hasValidationErrorOccurred());
}

TEST_CASE("sequences can handle subroutines", "[command]") {
    Tensor<int> tensor(getContext(), 8);
    Buffer<int> buffer(getContext(), 8);

    auto subA = createSubroutine(getContext(), simultaneous_use,
        clearTensor(tensor, { .data = 5 }));
    auto subB = createSubroutine(getContext(),
        clearTensor(tensor, { .size = 8, .data = 19 }),
        clearTensor(tensor, { .offset = 8, .size = 16, .data = 7 }));
    auto subC = createSubroutine(getContext(),
        clearTensor(tensor, { .offset = 24, .size = 8, .data = 23 }));

    REQUIRE(subA.simultaneousUse());
    REQUIRE(!subB.simultaneousUse());
    REQUIRE(!subC.simultaneousUse());

    beginSequence(getContext())
        .And(subA).And(retrieveTensor(tensor, buffer))
        .Then(subB).And(subC)
        .Then(retrieveTensor(tensor, buffer))
        .Submit();

    auto data = std::to_array<int>({ 19, 19, 7, 7, 7, 7, 23, 23 });
    REQUIRE(std::equal(data.begin(), data.end(), buffer.getMemory().data()));

    REQUIRE(!hasValidationErrorOccurred());
}
