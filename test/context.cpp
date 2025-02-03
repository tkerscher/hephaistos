#include <catch2/catch_test_macros.hpp>

#include <hephaistos/buffer.hpp>

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

TEST_CASE("resources can be destroyed invidually or all at once", "[context]") {
    REQUIRE(getResourceCount(getContext()) == 0);
    
    Buffer<int> b1(getContext(), 10);
    Buffer<int> b2(getContext(), 10);
    Buffer<int> b3(getContext(), 10);
    Buffer<int> b4(getContext(), 10);

    REQUIRE(getResourceCount(getContext()) == 4);

    Tensor<int> t1(getContext(), 10);
    Tensor<int> t2(getContext(), 10);
    Tensor<int> t3(getContext(), 10);
    Tensor<int> t4(getContext(), 10);

    REQUIRE(getResourceCount(getContext()) == 8);

    b1.destroy();
    b2.destroy();
    t3.destroy();
    
    REQUIRE(getResourceCount(getContext()) == 5);

    b1 = std::move(b4);
    b3 = std::move(b2);
    t1 = std::move(t3);
    t3 = std::move(t4);

    REQUIRE(getResourceCount(getContext()) == 3);

    destroyAllResources(getContext());

    REQUIRE(getResourceCount(getContext()) == 0);

    REQUIRE(!b1);
    REQUIRE(!b2);
    REQUIRE(!b3);
    REQUIRE(!b4);

    REQUIRE(!t1);
    REQUIRE(!t2);
    REQUIRE(!t3);
    REQUIRE(!t4);

    REQUIRE(!hasValidationErrorOccurred());
}

TEST_CASE("ResourceSnapshot can destroy newly created resources", "[context]") {
    Buffer<int> b1(getContext(), 10);
    Buffer<int> b2(getContext(), 10);
    Tensor<int> t1(getContext(), 10);
    Tensor<int> t2(getContext(), 10);

    ResourceSnapshot snapshot(getContext());
    snapshot.capture();

    REQUIRE(snapshot.count() == 4);
    REQUIRE(getResourceCount(getContext()) == 4);

    Buffer<int> b3(getContext(), 10);
    Buffer<int> b4(getContext(), 10);
    Tensor<int> t3(getContext(), 10);
    Tensor<int> t4(getContext(), 10);

    REQUIRE(snapshot.count() == 4);
    REQUIRE(getResourceCount(getContext()) == 8);

    snapshot.restore();

    REQUIRE(snapshot.count() == 4);
    REQUIRE(getResourceCount(getContext()) == 4);

    REQUIRE(b1);
    REQUIRE(b2);
    REQUIRE(!b3);
    REQUIRE(!b4);

    REQUIRE(t1);
    REQUIRE(t2);
    REQUIRE(!t3);
    REQUIRE(!t4);

    REQUIRE(!hasValidationErrorOccurred());
}
