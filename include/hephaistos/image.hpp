#pragma once

#include "hephaistos/buffer.hpp"
#include "hephaistos/command.hpp"
#include "hephaistos/context.hpp"
#include "hephaistos/handles.hpp"

//fwd
struct VkWriteDescriptorSet;

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

enum class AddressMode {
    REPEAT = 0,
    MIRRORED_REPEAT = 1,
    CLAMP_TO_EDGE = 2,
    //CLAMP_TO_BORDER = 3,
    MIRROR_CLAMP_TO_EDGE = 4
};

enum class Filter {
    NEAREST,
    LINEAR
};

[[nodiscard]] bool isFilterSupported(const DeviceHandle& device,
    ImageFormat format, Filter filter);
[[nodiscard]] bool isFilterSupported(const ContextHandle& context,
    ImageFormat format, Filter filter);

struct Sampler {
    AddressMode addressModeU = AddressMode::REPEAT;
    AddressMode addressModeV = AddressMode::REPEAT;
    AddressMode addressModeW = AddressMode::REPEAT;

    Filter filter = Filter::LINEAR;

    bool unnormalizedCoordinates = false;
};

class HEPHAISTOS_API Image : public Resource {
public:
    [[nodiscard]] uint32_t getWidth() const noexcept;
    [[nodiscard]] uint32_t getHeight() const noexcept;
    [[nodiscard]] uint32_t getDepth() const noexcept;
    [[nodiscard]] ImageFormat getFormat() const noexcept;
    [[nodiscard]] uint64_t size_bytes() const noexcept;

    void bindParameter(VkWriteDescriptorSet& binding) const;

    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;

    Image(Image&& other) noexcept;
    Image& operator=(Image&& other) noexcept;

    Image(ContextHandle context,
        ImageFormat format,
        uint32_t width,
        uint32_t height = 1,
        uint32_t depth = 1);
    virtual ~Image();

public: //internal
    const vulkan::Image& getImage() const noexcept;

private:
    ImageHandle image;
    ImageFormat format;
    uint32_t width, height, depth;

    struct Parameter;
    std::unique_ptr<Parameter> parameter;
};

class HEPHAISTOS_API Texture : public Resource {
public:
    [[nodiscard]] uint32_t getWidth() const noexcept;
    [[nodiscard]] uint32_t getHeight() const noexcept;
    [[nodiscard]] uint32_t getDepth() const noexcept;
    [[nodiscard]] ImageFormat getFormat() const noexcept;
    [[nodiscard]] uint64_t size_bytes() const noexcept;

    void bindParameter(VkWriteDescriptorSet& binding) const;

    Texture(const Texture&) = delete;
    Texture& operator=(const Texture&) = delete;

    Texture(Texture&& other) noexcept;
    Texture& operator=(Texture&& other) noexcept;

    Texture(ContextHandle context,
        ImageFormat format,
        uint32_t width,
        const Sampler& sampler = {});
    Texture(ContextHandle context,
        ImageFormat format,
        uint32_t width,
        uint32_t height,
        const Sampler& sampler = {});
    Texture(ContextHandle context,
        ImageFormat format,
        uint32_t width,
        uint32_t height,
        uint32_t depth,
        const Sampler& sampler = {});
    virtual ~Texture();

public: //internal
    const vulkan::Image& getImage() const noexcept;

private:
    ImageHandle image;
    ImageFormat format;
    uint32_t width, height, depth;

    struct Parameter;
    std::unique_ptr<Parameter> parameter;
};

class HEPHAISTOS_API ImageBuffer : public Buffer<Vec4<uint8_t>> {
public:
    using ElementType = Vec4<uint8_t>;
    static constexpr auto Format = ImageFormat::R8G8B8A8_UNORM;

public:
    [[nodiscard]] uint32_t getWidth() const;
    [[nodiscard]] uint32_t getHeight() const;

    [[nodiscard]] Image createImage(bool copy = true) const;
    [[nodiscard]] Texture createTexture(const Sampler& sampler = {}, bool copy = true) const;

    [[nodiscard]] static ImageBuffer load(ContextHandle context, const char* filename);
    [[nodiscard]] static ImageBuffer load(ContextHandle context, std::span<const std::byte> memory);

    void save(const char* filename) const;

    ImageBuffer(const ImageBuffer&) = delete;
    ImageBuffer& operator=(const ImageBuffer&) = delete;

    ImageBuffer(ImageBuffer&& other) noexcept;
    ImageBuffer& operator=(ImageBuffer&& other) noexcept;

    ImageBuffer(ContextHandle context, uint32_t width, uint32_t height);
    virtual ~ImageBuffer();

private:
    uint32_t width, height;
};

class HEPHAISTOS_API RetrieveImageCommand : public Command {
public:
    std::reference_wrapper<const Image> Source;
    std::reference_wrapper<const Buffer<std::byte>> Destination;

    virtual void record(vulkan::Command& cmd) const override;

    RetrieveImageCommand(const RetrieveImageCommand& other);
    RetrieveImageCommand& operator=(const RetrieveImageCommand& other);

    RetrieveImageCommand(RetrieveImageCommand&& other) noexcept;
    RetrieveImageCommand& operator=(RetrieveImageCommand&& other) noexcept;

    RetrieveImageCommand(const Image& src, const Buffer<std::byte>& dst);
    virtual ~RetrieveImageCommand();
};
[[nodiscard]] inline RetrieveImageCommand retrieveImage(
    const Image& src, const Buffer<std::byte>& dst)
{
    return RetrieveImageCommand(src, dst);
}

class HEPHAISTOS_API UpdateImageCommand : public Command {
public:
    std::reference_wrapper<const Buffer<std::byte>> Source;
    std::reference_wrapper<const Image> Destination;

    virtual void record(vulkan::Command& cmd) const override;

    UpdateImageCommand(const UpdateImageCommand& other);
    UpdateImageCommand& operator=(const UpdateImageCommand& other);

    UpdateImageCommand(UpdateImageCommand&& other) noexcept;
    UpdateImageCommand& operator=(UpdateImageCommand&& other) noexcept;

    UpdateImageCommand(const Buffer<std::byte>& src, const Image& dst);
    virtual ~UpdateImageCommand();
};
[[nodiscard]] inline UpdateImageCommand updateImage(
    const Buffer<std::byte>& src, const Image& dst)
{
    return UpdateImageCommand(src, dst);
}

class HEPHAISTOS_API UpdateTextureCommand : public Command {
public:
    std::reference_wrapper<const Buffer<std::byte>> Source;
    std::reference_wrapper<const Texture> Destination;

    virtual void record(vulkan::Command& cmd) const override;

    UpdateTextureCommand(const UpdateTextureCommand& other);
    UpdateTextureCommand& operator=(const UpdateTextureCommand& other);

    UpdateTextureCommand(UpdateTextureCommand&& other) noexcept;
    UpdateTextureCommand& operator=(UpdateTextureCommand&& other) noexcept;

    UpdateTextureCommand(const Buffer<std::byte>& src, const Texture& dst);
    virtual ~UpdateTextureCommand();
};
[[nodiscard]] inline UpdateTextureCommand updateTexture(
    const Buffer<std::byte>& src, const Texture& dst)
{
    return UpdateTextureCommand(src, dst);
}

}
