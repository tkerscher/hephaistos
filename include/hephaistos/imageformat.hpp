#pragma once

namespace hephaistos {

/**
 * @brief 2 channel Vector
*/
template<class T>
struct Vec2 {
    union {
        struct { T x, y; };
        struct { T r, g; };
    };
};

/**
 * @brief 4 channel vector
*/
template<class T>
struct Vec4 {
    union {
        struct { T x, y, z, w; };
        struct { T r, g, b, a; };
    };
};

/**
 * @brief Enumeration of supported image formats
*/
enum class ImageFormat {
    /**
     * @brief 8 bit per channel RGBA normalized to [0,1]
    */
    R8G8B8A8_UNORM = 37,
    /**
     * @brief 8 bit per channel RGBA normalized to [-1,1]
    */
    R8G8B8A8_SNORM = 38,
    /**
     * @brief 8 bit per channel RGBA stored using unsigned integer [0,255]
    */
    R8G8B8A8_UINT = 41,
    /**
     * @brief 8 bit per channel RGBA stored using signed integer [-128,127]
    */
    R8G8B8A8_SINT = 42,
    /**
     * @brief 16 bit per channel RGBA stored using unsigned integer [0,65535]
    */
    R16G16B16A16_UINT = 95,
    /**
     * @brief 16 bit per channel RGBA stored using signed integer [-32768,32767]
    */
    R16G16B16A16_SINT = 96,
    //R16G16B16A16_SFLOAT,
    /**
     * @brief 32 bit single channel stored using unsigned integers
    */
    R32_UINT = 98,
    /**
     * @brief 32 bit single channel stored using signed integers
    */
    R32_SINT = 99,
    /**
     * @brief 32 bit single channel stored using floats
    */
    R32_SFLOAT = 100,
    /**
     * @brief 32 bit per channel two channel image using unsigned integers
    */
    R32G32_UINT = 101,
    /**
     * @brief 32 bit per channel two channel image using signed integers
    */
    R32G32_SINT = 102,
    /**
     * @brief 32 bit per channel two channel image using floats
    */
    R32G32_SFLOAT = 103,
    /**
     * @brief 32 bit per channel RGBA image stored using unsigned integers
    */
    R32G32B32A32_UINT = 107,
    /**
     * @brief 32 bit per channel RGBA image stored using signed integers
    */
    R32G32B32A32_SINT = 108,
    /**
     * @brief 32 bit per channel RGBA image stored using floats
    */
    R32G32B32A32_SFLOAT = 109,
    /**
     * @brief placeholder for unknown/unsupported format encountered in reflection
    */
    UNKNOWN = 0x7FFFFFFF
};

/**
 * @brief Return the pixel size in bytes
 * 
 * @param format ImageFormat to query
*/
[[nodiscard]] HEPHAISTOS_API uint64_t getElementSize(ImageFormat format);

/**
 * @brief Image format trait storing element type of a single pixel
*/
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

/**
 * @brief Shorthand notation of ImageElementType::ElementType
*/
template<ImageFormat IF>
using ImageElementTypeValue = typename ImageElementType<IF>::ElementType;

}
