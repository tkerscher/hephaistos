#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>

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

}

TEST_CASE("timeline can be moved and destroyed", "[command]") {
    Timeline timeline(getContext());
    Timeline timeline2(std::move(timeline));

    REQUIRE(!timeline);
    REQUIRE(!timeline.getContext());
    REQUIRE(timeline2);
    REQUIRE(timeline2.getContext());

    //check moved timeline still works
    timeline2.setValue(10);
    REQUIRE(timeline2.getValue() == 10);

    timeline = std::move(timeline2);

    REQUIRE(!timeline2);
    REQUIRE(!timeline2.getContext());
    REQUIRE(timeline);
    REQUIRE(timeline.getContext());

    //check moved timeline still works
    timeline.setValue(20);
    REQUIRE(timeline.getValue() == 20);

    timeline.destroy();

    REQUIRE(!timeline);
    REQUIRE(!timeline.getContext());
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

TEST_CASE("sequences can wait on multiple timelines", "[command]") {
    Timeline timeline(getContext());
    Tensor<int> tensor(getContext(), 8);
    Buffer<int> buffer(getContext(), 8);
    auto mem = buffer.getMemory();
    std::memset(mem.data(), 0, 40);

    auto submission = beginSequence(getContext())
        .AndList(
            clearTensor(tensor, { .size = 8, .data = 19 }),
            clearTensor(tensor, { .offset = 8, .size = 16, .data = 7 }),
            clearTensor(tensor, { .offset = 24, .size = 8, .data = 23 })
        ).WaitFor(timeline, 5)
        .And(retrieveTensor(tensor, buffer)).Submit();

    REQUIRE(!submission.wait(100));
    REQUIRE(std::all_of(mem.begin(), mem.end(), [](int i) { return i == 0; }));

    timeline.setValue(3);
    REQUIRE(!submission.wait(100));
    REQUIRE(std::all_of(mem.begin(), mem.end(), [](int i) { return i == 0; }));

    timeline.setValue(5);
    REQUIRE(submission.wait(100));
    auto data = std::to_array<int>({ 19, 19, 7, 7, 7, 7, 23, 23 });
    REQUIRE(std::equal(mem.begin(), mem.end(), data.begin()));

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

TEST_CASE("subroutines can be moved and destroyed", "[command]") {
    Tensor<int> tensor(getContext(), 8);
    Buffer<int> buffer(getContext(), 8);

    auto sub = createSubroutine(getContext(), clearTensor(tensor, { .data = 42 }));
    Subroutine sub2(std::move(sub));

    REQUIRE(!sub);
    REQUIRE(!sub.getContext());
    REQUIRE(sub2);
    REQUIRE(sub2.getContext());

    sub = std::move(sub2);

    REQUIRE(!sub2);
    REQUIRE(!sub2.getContext());
    REQUIRE(sub);
    REQUIRE(sub.getContext());

    sub2 = std::move(sub);

    beginSequence(getContext()).And(sub2).Then(retrieveTensor(tensor, buffer)).Submit();

    auto data = std::to_array<int>({ 42, 42, 42, 42, 42, 42, 42, 42 });
    REQUIRE(std::equal(data.begin(), data.end(), buffer.getMemory().data()));

    sub2.destroy();

    REQUIRE(!sub2);
    REQUIRE(!sub2.getContext());

    REQUIRE(!hasValidationErrorOccurred());
}

TEST_CASE("one time submits can handle a list of commands", "[command]") {
    Tensor<int> tensor(getContext(), 8);
    Buffer<int> buffer(getContext(), 8);

    executeList(getContext(),
        clearTensor(tensor, { .size = 8, .data = 19 }),
        clearTensor(tensor, { .offset = 8, .size = 16, .data = 7 }),
        clearTensor(tensor, { .offset = 24, .size = 8, .data = 23 }),
        retrieveTensor(tensor, buffer));

    auto data = std::to_array<int>({ 19, 19, 7, 7, 7, 7, 23, 23 });
    REQUIRE(std::equal(data.begin(), data.end(), buffer.getMemory().data()));

    REQUIRE(!hasValidationErrorOccurred());
}
