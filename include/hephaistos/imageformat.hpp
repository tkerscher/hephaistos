#pragma once

namespace hephaistos {

template<class T>
struct Vec2 {
    union {
        struct { T x, y; };
        struct { T r, g; };
    };
};

template<class T>
struct Vec4 {
    union {
        struct { T x, y, z, w; };
        struct { T r, g, b, a; };
    };
};

enum class ImageFormat {
    R8G8B8A8_UNORM = 37,
    R8G8B8A8_SNORM = 38,
    R8G8B8A8_UINT = 41,
    R8G8B8A8_SINT = 42,
    R16G16B16A16_UINT = 95,
    R16G16B16A16_SINT = 96,
    //R16G16B16A16_SFLOAT,
    R32_UINT = 98,
    R32_SINT = 99,
    R32_SFLOAT = 100,
    R32G32_UINT = 101,
    R32G32_SINT = 102,
    R32G32_SFLOAT = 103,
    R32G32B32A32_UINT = 107,
    R32G32B32A32_SINT = 108,
    R32G32B32A32_SFLOAT = 109,
    UNKNOWN = 0x7FFFFFFF
};

[[nodiscard]] uint64_t getElementSize(ImageFormat format);

template<ImageFormat IF> struct ImageElementType {};

template<> struct ImageElementType<ImageFormat::R8G8B8A8_UNORM> { using ElementType = Vec4<uint8_t>; };
template<> struct ImageElementType<ImageFormat::R8G8B8A8_SNORM> { using ElementType = Vec4<int8_t>; };
template<> struct ImageElementType<ImageFormat::R8G8B8A8_UINT> { using ElementType = Vec4<uint8_t>; };
template<> struct ImageElementType<ImageFormat::R8G8B8A8_SINT> { using ElementType = Vec4<int8_t>; };

template<> struct ImageElementType<ImageFormat::R16G16B16A16_UINT> { using ElementType = Vec4<uint16_t>; };
template<> struct ImageElementType<ImageFormat::R16G16B16A16_SINT> { using ElementType = Vec4<int16_t>; };

template<> struct ImageElementType<ImageFormat::R32_UINT> { using ElementType = uint32_t; };
template<> struct ImageElementType<ImageFormat::R32_SINT> { using ElementType = int32_t; };
template<> struct ImageElementType<ImageFormat::R32_SFLOAT> { using ElementType = float; };

template<> struct ImageElementType<ImageFormat::R32G32_UINT> { using ElementType = Vec2<uint32_t>; };
template<> struct ImageElementType<ImageFormat::R32G32_SINT> { using ElementType = Vec2<int32_t>; };
template<> struct ImageElementType<ImageFormat::R32G32_SFLOAT> { using ElementType = Vec2<float>; };

template<> struct ImageElementType<ImageFormat::R32G32B32A32_UINT> { using ElementType = Vec4<uint32_t>; };
template<> struct ImageElementType<ImageFormat::R32G32B32A32_SINT> { using ElementType = Vec4<int32_t>; };
template<> struct ImageElementType<ImageFormat::R32G32B32A32_SFLOAT> { using ElementType = Vec4<float>; };

template<ImageFormat IF>
using ImageElementTypeValue = typename ImageElementType<IF>::ElementType;

}
