#pragma once

#include "hephaistos/argument.hpp"
#include "hephaistos/buffer.hpp"
#include "hephaistos/command.hpp"
#include "hephaistos/context.hpp"
#include "hephaistos/handles.hpp"
#include "hephaistos/imageformat.hpp"

namespace hephaistos {

/**
 * @brief List of behaviour for sampling Texture out of bounds
*/
enum class AddressMode {
    /**
     * @brief Repeats the texture
    */
    REPEAT = 0,
    /**
     * @brief Repeats the texture after mirroring it
    */
    MIRRORED_REPEAT = 1,
    /**
     * @brief Repeats the last pixel in the specific direction
    */
    CLAMP_TO_EDGE = 2,
    //CLAMP_TO_BORDER = 3,
    /**
     * @brief Repeats the last pixel in the opposite direction
    */
    MIRROR_CLAMP_TO_EDGE = 4
};

/**
 * @brief List of methods to interpolate in between pixels
*/
enum class Filter {
    /**
     * @brief Clamps to the nearest pixel
    */
    NEAREST,
    /**
     * @brief Linear interpolation
    */
    LINEAR
};

/**
 * @brief Queries wether the given combination of format and filter is supported
 * 
 * @param device Device on which to query support
 * @param format Image format to query
 * @param filter Filter method
 * 
 * @return True, if the combination is supported by the device, false otherwise
*/
[[nodiscard]] bool isFilterSupported(const DeviceHandle& device,
    ImageFormat format, Filter filter);
/**
 * @brief Queries wether the given combination of format and filter is supported
 * 
 * @param context Context on which to query support
 * @param format Image format to query
 * @param filter Filter method
 * 
 * @return True, if the combination is supported by the context, false otherwise
*/
[[nodiscard]] bool isFilterSupported(const ContextHandle& context,
    ImageFormat format, Filter filter);

/**
 * @brief Configuration of a sampler
 * 
 * Samplers are used to lookup pixels in a texture using more advanced methods
 * like interpolation and normalized coordinates.
*/
struct Sampler {
    /**
     * @brief Address mode in U dimension
    */
    AddressMode addressModeU = AddressMode::REPEAT;
    /**
     * @brief Address mode in V dimension
    */
    AddressMode addressModeV = AddressMode::REPEAT;
    /**
     * @brief Address mode in W dimension
    */
    AddressMode addressModeW = AddressMode::REPEAT;

    /**
     * @brief Filter method
    */
    Filter filter = Filter::LINEAR;

    /**
     * @brief Wether to use unnormalized coordinates
     * 
     * If false, pixel coordinates are in the range [0,1], otherwise whole
     * numbers represents whole pixels.
    */
    bool unnormalizedCoordinates = false;
};

/**
 * @brief Storage image allocated on device memory
 * 
 * Storage images are allocated from device memory and can be read and written
 * pixel-wise inside programs after binding them. They are stored in a device
 * specific memory layout, which may result in improved performance compared to
 * regular tensor.
*/
class HEPHAISTOS_API Image : public Argument, public Resource {
public:
    /**
     * @brief Width of the image in pixels
    */
    [[nodiscard]] uint32_t getWidth() const noexcept;
    /**
     * @brief Height of the image in pixels
    */
    [[nodiscard]] uint32_t getHeight() const noexcept;
    /**
     * @brief Depth of the image in pixels
    */
    [[nodiscard]] uint32_t getDepth() const noexcept;
    /**
     * @brief Format of the image
    */
    [[nodiscard]] ImageFormat getFormat() const noexcept;
    /**
     * @brief Size of the image in linear memory layout in bytes 
     * 
     * @note This is not the size the image takes on the device, but can be used
     *       for allocating memory on the host to retrieve or stage the image.
    */
    [[nodiscard]] uint64_t size_bytes() const noexcept;

    void bindParameter(VkWriteDescriptorSet& binding) const final override;

    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;

    Image(Image&& other) noexcept;
    Image& operator=(Image&& other) noexcept;

    /**
     * @brief Allocates an image on the given context
     * 
     * @param context Context onto which to create the image
     * @param format Format of the image to create
     * @param width Width of the image in pixels
     * @param height Height of the image in pixels
     * @param depth Depth of the image in pixels
    */
    Image(ContextHandle context,
        ImageFormat format,
        uint32_t width,
        uint32_t height = 1,
        uint32_t depth = 1);
    ~Image() override;

public: //internal
    const vulkan::Image& getImage() const noexcept;

private:
    ImageHandle image;
    ImageFormat format;
    uint32_t width, height, depth;

    struct Parameter;
    std::unique_ptr<Parameter> parameter;
};

/**
 * @brief Texture allocated on device memory
 * 
 * Textures are readonly and optimized for lookups providing more advanced
 * methods like interpolation. Texture lookups may be more performant than
 * tensor or image lookups depending on the workload.
*/
class HEPHAISTOS_API Texture : public Argument, public Resource {
public:
    /**
     * @brief Width of the texture in pixels
    */
    [[nodiscard]] uint32_t getWidth() const noexcept;
    /**
     * @brief Height of the texture in pixel
    */
    [[nodiscard]] uint32_t getHeight() const noexcept;
    /**
     * @brief Depth of the texture in pixel
    */
    [[nodiscard]] uint32_t getDepth() const noexcept;
    /**
     * @brief Format of the texture
    */
    [[nodiscard]] ImageFormat getFormat() const noexcept;
    /**
     * @brief Size of the texture in bytes
     * 
     * @note This is not the size the image takes on the device, but can be used
     *       for allocating memory on the host to retrieve or stage the image.
    */
    [[nodiscard]] uint64_t size_bytes() const noexcept;

    void bindParameter(VkWriteDescriptorSet& binding) const final override;

    Texture(const Texture&) = delete;
    Texture& operator=(const Texture&) = delete;

    Texture(Texture&& other) noexcept;
    Texture& operator=(Texture&& other) noexcept;

    /**
     * @brief Allocates a texture on the given context
     * 
     * @param context Context onto which to create the texture
     * @param format Image format of the texture
     * @param width Width of the texture in pixels
     * @param sampler Sampler configuration used for texture lookups
    */
    Texture(ContextHandle context,
        ImageFormat format,
        uint32_t width,
        const Sampler& sampler = {});
    /**
     * @brief Allocates a texture on the given context
     * 
     * @param context Context onto which to create the texture
     * @param format Image format of the texture
     * @param width Width of the texture in pixels
     * @param height Height of the texture in pixels
     * @param sampler Sampler configuration used for texture lookups
    */   
    Texture(ContextHandle context,
        ImageFormat format,
        uint32_t width,
        uint32_t height,
        const Sampler& sampler = {});
    /**
     * @brief Allocates a texture on the given context
     * 
     * @param context Context onto which to create the texture
     * @param format Image format of the texture
     * @param width Width of the texture in pixels
     * @param height Height of the texture in pixels
     * @param depth Depth of the texture in pixels
     * @param sampler Sampler configuration used for texture lookups
    */
    Texture(ContextHandle context,
        ImageFormat format,
        uint32_t width,
        uint32_t height,
        uint32_t depth,
        const Sampler& sampler = {});
    ~Texture() override;

public: //internal
    const vulkan::Image& getImage() const noexcept;

private:
    ImageHandle image;
    ImageFormat format;
    uint32_t width, height, depth;

    struct Parameter;
    std::unique_ptr<Parameter> parameter;
};

/**
 * @brief Image buffer allocated on host memory
 * 
 * Image buffer is a utility class allocating enough memory to store 2D RGBA
 * images * in linear memory layout providing additional methods for loading
 * and saving images from disk.
*/
class HEPHAISTOS_API ImageBuffer : public Buffer<Vec4<uint8_t>> {
public:
    /**
     * @brief Type of a single pixel
    */
    using ElementType = Vec4<uint8_t>;
    /**
     * @brief Format of the image
    */
    static constexpr auto Format = ImageFormat::R8G8B8A8_UNORM;

public:
    /**
     * @brief Width of the image in pixels
    */
    [[nodiscard]] uint32_t getWidth() const;
    /**
     * @brief Height of the image in pixels
    */
    [[nodiscard]] uint32_t getHeight() const;

    /**
     * @brief Creates an Image of equal dimension
     * 
     * @param copy If true, copies the content of this buffer to the newly
     *             created image
     * @return Newly created Image
    */
    [[nodiscard]] Image createImage(bool copy = true) const;
    /**
     * @brief Creates a Texture of equal dimension
     * 
     * @param sample Sampler to use with the texture
     * @param copy If true, copies the content of this buffer to the newly
     *             created texture
     * @return Newly created Texture
    */
    [[nodiscard]] Texture createTexture(const Sampler& sampler = {}, bool copy = true) const;

    /**
     * @brief Loads image from disk
     * 
     * Loads the image at the given filepath from disk and returns its contents
     * in a new ImageBuffer.
     * 
     * @param context Context onto which to create the ImageBuffer
     * @param filename Path to the image on disk to load
     * @return New ImageBuffer containing the loaded image
    */
    [[nodiscard]] static ImageBuffer load(ContextHandle context, const char* filename);
    /**
     * @brief Loads image from memory
     * 
     * Loads the image stored in the given memory range in a serialized format
     * and returns the deserialize image in a new ImageBuffer.
     * 
     * @param context Context onto which to create the ImageBuffer
     * @param memory Memory range containing the serialized image
     * @return New ImageBuffer containing the deserialized image
    */
    [[nodiscard]] static ImageBuffer load(ContextHandle context, std::span<const std::byte> memory);

    /**
     * @brief Saves the current content if the ImageBuffer at the given filepath
     * 
     * @param filename Filepath to where to save the image
    */
    void save(const char* filename) const;

    ImageBuffer(const ImageBuffer&) = delete;
    ImageBuffer& operator=(const ImageBuffer&) = delete;

    ImageBuffer(ImageBuffer&& other) noexcept;
    ImageBuffer& operator=(ImageBuffer&& other) noexcept;

    /**
     * @brief Creates a new ImageBuffer
     * 
     * @param context Context onto which to create the ImageBuffer
     * @param width Width of the image in pixels
     * @param height Height of the image in pixels
    */
    ImageBuffer(ContextHandle context, uint32_t width, uint32_t height);
    ~ImageBuffer() override;

private:
    uint32_t width, height;
};

/**
 * @brief Command for retrieving an image from device to host
*/
class HEPHAISTOS_API RetrieveImageCommand : public Command {
public:
    /**
     * @brief Image source to copy from
    */
    std::reference_wrapper<const Image> Source;
    /**
     * @brief Buffer destination to copy to
    */
    std::reference_wrapper<const Buffer<std::byte>> Destination;

    void record(vulkan::Command& cmd) const override;

    RetrieveImageCommand(const RetrieveImageCommand& other);
    RetrieveImageCommand& operator=(const RetrieveImageCommand& other);

    RetrieveImageCommand(RetrieveImageCommand&& other) noexcept;
    RetrieveImageCommand& operator=(RetrieveImageCommand&& other) noexcept;

    /**
     * @brief Creates a new RetrieveImageCommand
     * 
     * @param src Image source to copy from
     * @param dst Buffer destination to copy to
    */
    RetrieveImageCommand(const Image& src, const Buffer<std::byte>& dst);
    ~RetrieveImageCommand() override;
};
/**
 * @brief Creates a RetrieveImageCommand for copying an image from device to host
 * 
 * @param src Image source to copy from
 * @param dst Buffer destination to copy to
*/
[[nodiscard]] inline RetrieveImageCommand retrieveImage(
    const Image& src, const Buffer<std::byte>& dst)
{
    return RetrieveImageCommand(src, dst);
}

/**
 * @brief Command for uploading image data from host to device
*/
class HEPHAISTOS_API UpdateImageCommand : public Command {
public:
    /**
     * @brief Source buffer to copy from
    */
    std::reference_wrapper<const Buffer<std::byte>> Source;
    /**
     * @brief Destination image to copy to
    */
    std::reference_wrapper<const Image> Destination;

    void record(vulkan::Command& cmd) const override;

    UpdateImageCommand(const UpdateImageCommand& other);
    UpdateImageCommand& operator=(const UpdateImageCommand& other);

    UpdateImageCommand(UpdateImageCommand&& other) noexcept;
    UpdateImageCommand& operator=(UpdateImageCommand&& other) noexcept;

    /**
     * @brief Creates a new UpdateImageCommand
     * 
     * @param src Buffer source to copy from
     * @param dst Image destination to copy to
    */
    UpdateImageCommand(const Buffer<std::byte>& src, const Image& dst);
    ~UpdateImageCommand() override;
};
/**
 * @brief Creates a UpdateImageCommand for uploading an image from host to device
 * 
 * @param src Buffer source to copy from
 * @param dst Image destination to copy to
*/
[[nodiscard]] inline UpdateImageCommand updateImage(
    const Buffer<std::byte>& src, const Image& dst)
{
    return UpdateImageCommand(src, dst);
}

/**
 * @brief Command for uploading a texture from host to device
*/
class HEPHAISTOS_API UpdateTextureCommand : public Command {
public:
    /**
     * @brief Buffer source to copy from
    */
    std::reference_wrapper<const Buffer<std::byte>> Source;
    /**
     * @brief Texture destination to copy to
    */
    std::reference_wrapper<const Texture> Destination;

    void record(vulkan::Command& cmd) const override;

    UpdateTextureCommand(const UpdateTextureCommand& other);
    UpdateTextureCommand& operator=(const UpdateTextureCommand& other);

    UpdateTextureCommand(UpdateTextureCommand&& other) noexcept;
    UpdateTextureCommand& operator=(UpdateTextureCommand&& other) noexcept;

    /**
     * @brief Creates a new UpdateTextureCommand
     * 
     * @param src Buffer source to copy from
     * @param dst Texture destination to copy to
    */
    UpdateTextureCommand(const Buffer<std::byte>& src, const Texture& dst);
    ~UpdateTextureCommand() override;
};
/**
 * @brief Creates a UpdateTextureCommand for uploading a texture from host to device
 * 
 * @param src Buffer source to copy from
 * @param dst Texture destination to copy to
*/
[[nodiscard]] inline UpdateTextureCommand updateTexture(
    const Buffer<std::byte>& src, const Texture& dst)
{
    return UpdateTextureCommand(src, dst);
}

}
