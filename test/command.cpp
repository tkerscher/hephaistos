#include <catch2/catch_test_macros.hpp>

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

//TODO: Fix this
//TEST_CASE("sequences can wait for the cpu", "[command]") {
//    Timeline timeline(getContext());
//
//    auto submission = beginSequence(timeline).WaitFor(5).Submit();
//
//    REQUIRE(!submission.wait(100));
//    timeline.setValue(2);
//    REQUIRE(!submission.wait(100));
//    timeline.setValue(6);
//    REQUIRE(submission.wait(100));
//
//    REQUIRE(!hasValidationErrorOccurred());
//}
