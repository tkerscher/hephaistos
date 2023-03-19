#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <algorithm>
#include <array>
#include <span>

#include <hephaistos/command.hpp>
#include <hephaistos/image.hpp>

using namespace hephaistos;

namespace {

ContextHandle getContext() {
    static ContextHandle context = createEmptyContext();
    if (!context)
        context = createContext();
    return context;
}

std::array<uint8_t, 36> data = { {
    0,1,2,3,    4,54,12,200,  45,12,99,102,
    0,123,0,45, 78,50,101,56, 22,23,89,7,
    12,0,56,0,  0,45,9,9,     12,21,3,78
} };

std::array<int32_t, 16> data2 = { {
    12, 7893, 0, 132312,
    456, 12346, 78516, 13,
    456, 73561, 4286, 7802,
    705, 46, 305, 334
} };

}

TEST_CASE("images know their size", "[image]") {
    auto format = GENERATE(as<ImageFormat>{},
        ImageFormat::R8G8B8A8_UNORM,
        ImageFormat::R8G8B8A8_SNORM,
        ImageFormat::R8G8B8A8_UINT,
        ImageFormat::R8G8B8A8_SINT,
        ImageFormat::R16G16B16A16_UINT,
        ImageFormat::R16G16B16A16_SINT,
        ImageFormat::R32_UINT,
        ImageFormat::R32_SINT,
        ImageFormat::R32_SFLOAT,
        ImageFormat::R32G32_UINT,
        ImageFormat::R32G32_SINT,
        ImageFormat::R32G32_SFLOAT,
        ImageFormat::R32G32B32A32_UINT,
        ImageFormat::R32G32B32A32_SINT,
        ImageFormat::R32G32B32A32_SFLOAT
    );
    auto width = GENERATE(1, 32, 80);
    auto height = GENERATE(1, 5, 32);
    auto depth = GENERATE(1, 16, 60);
    Image image(getContext(), format, width, height, depth);

    //check if settings are correct
    REQUIRE(image.getWidth() == width);
    REQUIRE(image.getHeight() == height);
    REQUIRE(image.getDepth() == depth);
    REQUIRE(image.getFormat() == format);

    //check size
    auto size = getElementSize(format) * width * height * depth;
    REQUIRE(image.size_bytes() == size);

    REQUIRE(!hasValidationErrorOccurred());
}

TEST_CASE("image buffer know their size", "[image]") {
    auto width = GENERATE(1, 24, 60);
    auto height = GENERATE(1, 12, 32);
    ImageBuffer buffer(getContext(), width, height);

    REQUIRE(buffer.getWidth() == width);
    REQUIRE(buffer.getHeight() == height);

    REQUIRE(!hasValidationErrorOccurred());
}

TEST_CASE("image buffer can create images", "[image]") {
    auto width = GENERATE(1, 24, 60);
    auto height = GENERATE(1, 12, 32);
    ImageBuffer buffer(getContext(), width, height);

    SECTION("image has same dimension") {
        auto image = buffer.createImage(false);
        REQUIRE(image.getWidth() == buffer.getWidth());
        REQUIRE(image.getHeight() == buffer.getHeight());
    }

    SECTION("image has correct format") {
        auto image = buffer.createImage(false);
        auto size = getElementSize(ImageBuffer::Format) * width * height;

        REQUIRE(image.getFormat() == ImageBuffer::Format);
        REQUIRE(image.size_bytes() == size);
    }

    REQUIRE(!hasValidationErrorOccurred());
}

TEST_CASE("image buffer can populate created image", "[image]") {
    ImageBuffer buffer(getContext(), 3, 3);

    //fill data
    std::memcpy(buffer.getMemory().data(), data.data(), 36);
    //create copy
    auto image = buffer.createImage(true);

    //Fetch data back
    ImageBuffer bufferOut(getContext(), 3, 3);
    Timeline timeline(getContext());
    beginSequence(timeline)
        .And(retrieveImage(image, bufferOut))
        .Submit();
    
    //compare data
    auto outMemory = std::span<uint8_t>{
        reinterpret_cast<uint8_t*>(bufferOut.getMemory().data()), 36
    };
    REQUIRE(std::equal(data.begin(), data.end(), outMemory.begin(), outMemory.end()));

    REQUIRE(!hasValidationErrorOccurred());
}

TEST_CASE("images can be copied to and from gpu", "[image]") {
    Buffer<int32_t> bufferIn(getContext(), data2);
    Buffer<int32_t> bufferOut(getContext(), data2.size());
    Image image(getContext(), ImageFormat::R32_SINT, 4, 4);

    //copy data
    Timeline timeline(getContext());
    beginSequence(timeline)
        .And(updateImage(bufferIn, image))
        .Then(retrieveImage(image, bufferOut))
        .Submit();
    
    //check data
    auto mem = bufferOut.getMemory();
    REQUIRE(std::equal(data2.begin(), data2.end(), mem.begin(), mem.end()));

    REQUIRE(!hasValidationErrorOccurred());
}
