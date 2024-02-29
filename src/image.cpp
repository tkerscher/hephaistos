#include "hephaistos/image.hpp"

#include "volk.h"
#include "vk_mem_alloc.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "vk/result.hpp"
#include "vk/types.hpp"
#include "vk/util.hpp"

namespace hephaistos {

/********************************** FORMAT ************************************/

#define ELEMENT_SIZE(x) \
case ImageFormat::x: \
    return sizeof(ImageElementTypeValue<ImageFormat::x>);

uint64_t getElementSize(ImageFormat format) {
    switch (format) {
        ELEMENT_SIZE(R8G8B8A8_UNORM)
        ELEMENT_SIZE(R8G8B8A8_SNORM)
        ELEMENT_SIZE(R8G8B8A8_UINT)
        ELEMENT_SIZE(R8G8B8A8_SINT)
        ELEMENT_SIZE(R16G16B16A16_UINT)
        ELEMENT_SIZE(R16G16B16A16_SINT)
        ELEMENT_SIZE(R32_UINT)
        ELEMENT_SIZE(R32_SINT)
        ELEMENT_SIZE(R32_SFLOAT)
        ELEMENT_SIZE(R32G32_UINT)
        ELEMENT_SIZE(R32G32_SINT)
        ELEMENT_SIZE(R32G32_SFLOAT)
        ELEMENT_SIZE(R32G32B32A32_UINT)
        ELEMENT_SIZE(R32G32B32A32_SINT)
        ELEMENT_SIZE(R32G32B32A32_SFLOAT)
    default:
        throw std::runtime_error("Unknown image format");
    }
}

#undef ELEMENT_SIZE

/*********************************** IMAGE ************************************/

struct Image::Parameter {
    VkDescriptorImageInfo info;
};

uint32_t Image::getWidth() const noexcept { return width; } 
uint32_t Image::getHeight() const noexcept { return height; } 
uint32_t Image::getDepth() const noexcept { return depth; } 
ImageFormat Image::getFormat() const noexcept { return format; }
const vulkan::Image& Image::getImage() const noexcept { return *image; }
uint64_t Image::size_bytes() const noexcept {
    return getElementSize(format) * width * height * depth;
}

void Image::bindParameter(VkWriteDescriptorSet& binding) const {
    binding.pNext            = nullptr;
    binding.pImageInfo       = &parameter->info;
    binding.pTexelBufferView = nullptr;
    binding.pBufferInfo      = nullptr;
}

Image::Image(Image&& other) noexcept
    : Resource(std::move(other))
    , image(std::move(image))
    , parameter(std::move(other.parameter))
    , format(other.format)
    , width(other.width)
    , height(other.height)
    , depth(other.depth)
{}
Image& Image::operator=(Image&& other) noexcept {
    Resource::operator=(std::move(other));
    image = std::move(other.image);
    parameter = std::move(other.parameter);
    format = other.format;
    width = other.width;
    height = other.height;
    depth = other.depth;
    return *this;
}

Image::Image(ContextHandle context, ImageFormat format,
    uint32_t width, uint32_t height, uint32_t depth
)
    : Resource(std::move(context))
    , image(vulkan::createImage(
        getContext(),
        static_cast<VkFormat>(format),
        width, height, depth,
        VK_IMAGE_USAGE_STORAGE_BIT |
        VK_IMAGE_USAGE_TRANSFER_DST_BIT |
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT
    ))
    , parameter(std::make_unique<Parameter>())
    , format(format)
    , width(width)
    , height(height)
    , depth(depth)
{
    //transition image from undefined to general
    auto& con = *getContext();
    oneTimeSubmit(con, [&image = image->image, &con](VkCommandBuffer cmdBuffer) {
        VkImageMemoryBarrier barrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .image = image,
            .subresourceRange = VkImageSubresourceRange{
                VK_IMAGE_ASPECT_COLOR_BIT,
                0, 1, 0, 1
            }
        };
        con.fnTable.vkCmdPipelineBarrier(cmdBuffer,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_DEPENDENCY_BY_REGION_BIT,
            0, nullptr,
            0, nullptr,
            1, &barrier);
    });

    parameter->info = VkDescriptorImageInfo{
        .imageView = image->view,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL
    };
}
Image::~Image() = default;

/********************************** TEXTURE ***********************************/

bool isFilterSupported(
    const DeviceHandle& device,
    ImageFormat format,
    Filter filter)
{
    VkFormatProperties props;
    vkGetPhysicalDeviceFormatProperties(device->device,
        static_cast<VkFormat>(format), &props);

    VkFormatFeatureFlags flags;
    switch (filter) {
    case hephaistos::Filter::NEAREST:
        flags = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
        break;
    case hephaistos::Filter::LINEAR:
        flags = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
                VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
        break;
    default:
        throw std::logic_error("Unknown filter");
    }

    return (props.optimalTilingFeatures & flags) == flags;
}
bool isFilterSupported(
    const ContextHandle& context,
    ImageFormat format,
    Filter filter)
{
    return isFilterSupported(getDevice(context), format, filter);
}

struct Texture::Parameter {
    VkSampler sampler;
    VkDescriptorImageInfo info;
};

uint32_t Texture::getWidth() const noexcept { return width; }
uint32_t Texture::getHeight() const noexcept { return height; }
uint32_t Texture::getDepth() const noexcept { return depth; }
ImageFormat Texture::getFormat() const noexcept { return format; }
const vulkan::Image& Texture::getImage() const noexcept { return *image; }
uint64_t Texture::size_bytes() const noexcept {
    return getElementSize(format) * width * height * depth;
}

void Texture::bindParameter(VkWriteDescriptorSet& binding) const {
    binding.pNext = nullptr;
    binding.pImageInfo = &parameter->info;
    binding.pTexelBufferView = nullptr;
    binding.pBufferInfo = nullptr;
}

Texture::Texture(Texture&& other) noexcept
    : Resource(std::move(other))
    , image(std::move(image))
    , parameter(std::move(other.parameter))
    , format(other.format)
    , width(other.width)
    , height(other.height)
    , depth(other.depth)
{}
Texture& Texture::operator=(Texture&& other) noexcept {
    Resource::operator=(std::move(other));
    image = std::move(other.image);
    parameter = std::move(other.parameter);
    format = other.format;
    width = other.width;
    height = other.height;
    depth = other.depth;
    return *this;
}

Texture::Texture(
    ContextHandle context,
    ImageFormat format,
    uint32_t width,
    const Sampler& sampler
)
    : Texture(std::move(context), format, width, 1, 1, sampler)
{}
Texture::Texture(
    ContextHandle context,
    ImageFormat format,
    uint32_t width,
    uint32_t height,
    const Sampler& sampler
)
    : Texture(std::move(context), format, width, height, 1, sampler)
{}
Texture::Texture(
    ContextHandle context,
    ImageFormat format,
    uint32_t width,
    uint32_t height,
    uint32_t depth,
    const Sampler& sampler
)
    : Resource(std::move(context))
    , image(vulkan::createImage(
        getContext(),
        static_cast<VkFormat>(format),
        width, height, depth,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT |
        VK_IMAGE_USAGE_SAMPLED_BIT
    ))
    , parameter(std::make_unique<Parameter>())
    , format(format)
    , width(width)
    , height(height)
    , depth(depth)
{
    //create sampler
    VkSamplerCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = static_cast<VkFilter>(sampler.filter),
        .minFilter = static_cast<VkFilter>(sampler.filter),
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR, //Really Needed?
        .addressModeU = static_cast<VkSamplerAddressMode>(sampler.addressModeU),
        .addressModeV = static_cast<VkSamplerAddressMode>(sampler.addressModeV),
        .addressModeW = static_cast<VkSamplerAddressMode>(sampler.addressModeW),
        .unnormalizedCoordinates = sampler.unnormalizedCoordinates
    };
    vulkan::checkResult(getContext()->fnTable.vkCreateSampler(
        getContext()->device, &info, nullptr, &parameter->sampler));

    parameter->info = VkDescriptorImageInfo{
        .sampler = parameter->sampler,
        .imageView = image->view,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    };
}

Texture::~Texture() {
    auto& con = *getContext();
    con.fnTable.vkDestroySampler(con.device, parameter->sampler, nullptr);
}

/******************************** IMAGE BUFFER ********************************/

uint32_t ImageBuffer::getWidth() const { return width; }
uint32_t ImageBuffer::getHeight() const { return height; }

Image ImageBuffer::createImage(bool copy) const {
    Image result(getContext(), Format, width, height, 1);

    //copy image via one time submit -> thus synchronous
    if (copy) {
        UpdateImageCommand copyCommand(*this, result);
        vulkan::oneTimeSubmit(*getContext(), [&copyCommand](VkCommandBuffer cmd) {
            //package into vulkan::command
            vulkan::Command command{ cmd, 0 };
            copyCommand.record(command);
        });
    }

    return result;
}

Texture ImageBuffer::createTexture(const Sampler& sampler, bool copy) const {
    Texture result(getContext(), Format, width, height, sampler);

    //copy image via one time submit -> synchronous
    if (copy) {
        UpdateTextureCommand copyCommand(*this, result);
        vulkan::oneTimeSubmit(*getContext(), [&copyCommand](VkCommandBuffer cmd) {
            //package into vulkan::command
            vulkan::Command command{ cmd, 0 };
            copyCommand.record(command);
        });
    }

    return result;
}

ImageBuffer ImageBuffer::load(ContextHandle context, const char* filename) {
    //load image
    //unfortunately, stbi wants to allocate its own memory, so we have an extra copy...
    int x, y, n;
    unsigned char* data = stbi_load(filename, &x, &y, &n, STBI_rgb_alpha);
    ImageBuffer result(std::move(context), x, y);
    auto mem = result.getMemory();

    //copy image and free data
    memcpy(mem.data(), data, result.size_bytes());
    stbi_image_free(data);

    //done
    return result;
}
ImageBuffer ImageBuffer::load(ContextHandle context, std::span<const std::byte> memory) {
    int x, y, n;
    auto data = stbi_load_from_memory(
        reinterpret_cast<const unsigned char*>(memory.data()),
        static_cast<int>(memory.size_bytes()),
        &x, &y, &n, STBI_rgb_alpha);
    ImageBuffer result(context, x, y);
    auto mem = result.getMemory();

    memcpy(mem.data(), data, mem.size_bytes());
    stbi_image_free(data);

    return result;
}

void ImageBuffer::save(const char* filename) const {
    stbi_write_png(filename,
        getWidth(), getHeight(), STBI_rgb_alpha,
        getMemory().data(), getWidth() * 4);
}

ImageBuffer::ImageBuffer(ImageBuffer&& other) noexcept
    : Buffer<ElementType>(std::move(other))
    , width(other.width)
    , height(other.height)
{}
ImageBuffer& ImageBuffer::operator=(ImageBuffer&& other) noexcept {
    Buffer<ElementType>::operator=(std::move(other));
    width = other.width;
    height = other.height;
    return *this;
}

ImageBuffer::ImageBuffer(ContextHandle context, uint32_t width, uint32_t height)
    : Buffer<ElementType>(std::move(context), width * height)
    , width(width)
    , height(height)
{}
ImageBuffer::~ImageBuffer() = default;

/************************************ COPY ************************************/

namespace {

constexpr auto DIFFERENT_CONTEXT_ERROR_STR =
    "Source and destination of a copy command must originate from the same context!";
constexpr auto SIZE_MISMATCH_ERROR_STR =
    "Source and destination must have the same size!";

}

void RetrieveImageCommand::record(vulkan::Command& cmd) const {
    //alias for shorter code
    auto& src = Source.get();
    auto& dst = Destination.get();
    //Check for src and dst to be from the same context
    if (src.getContext().get() != dst.getContext().get())
        throw std::logic_error(DIFFERENT_CONTEXT_ERROR_STR);
    auto& context = src.getContext();
    //check both have the same size
    if (src.size_bytes() != dst.size_bytes())
        throw std::logic_error(SIZE_MISMATCH_ERROR_STR);
    auto size = src.size_bytes();

    //we're acting on the transfer stage
    cmd.stage |= VK_PIPELINE_STAGE_TRANSFER_BIT;

    //ensure writing to image finished and prepare image as transfer src
    VkImageMemoryBarrier barrier{
        .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask    = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask    = VK_ACCESS_TRANSFER_READ_BIT,
        .oldLayout        = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout        = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .image            = src.getImage().image,
        .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
    };
    context->fnTable.vkCmdPipelineBarrier(cmd.buffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_DEPENDENCY_BY_REGION_BIT,
        0, nullptr,
        0, nullptr,
        1, &barrier);

    //issue copy
    VkBufferImageCopy copy{
        .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0 ,1 },
        .imageExtent = { src.getWidth(), src.getHeight(), src.getDepth() }
    };
    context->fnTable.vkCmdCopyImageToBuffer(cmd.buffer,
        src.getImage().image,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        dst.getBuffer().buffer,
        1, &copy);

    //ensure transfer finished and return image layout
    barrier = VkImageMemoryBarrier{
        .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask    = VK_ACCESS_TRANSFER_READ_BIT,
        .dstAccessMask    = VK_ACCESS_MEMORY_WRITE_BIT, //TODO: Not entirely shure about that one
        .oldLayout        = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .newLayout        = VK_IMAGE_LAYOUT_GENERAL,
        .image            = src.getImage().image,
        .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
    };
    VkBufferMemoryBarrier bufferBarrier{
        .sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
        .buffer        = dst.getBuffer().buffer,
        .size          = VK_WHOLE_SIZE
    };
    context->fnTable.vkCmdPipelineBarrier(cmd.buffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT,
        VK_DEPENDENCY_BY_REGION_BIT,
        0, nullptr,
        1, &bufferBarrier,
        1, &barrier);
}

RetrieveImageCommand::RetrieveImageCommand(const RetrieveImageCommand& other) = default;
RetrieveImageCommand& RetrieveImageCommand::operator=(const RetrieveImageCommand& other) = default;

RetrieveImageCommand::RetrieveImageCommand(RetrieveImageCommand&& other) noexcept = default;
RetrieveImageCommand& RetrieveImageCommand::operator=(RetrieveImageCommand&& other) noexcept = default;

RetrieveImageCommand::RetrieveImageCommand(const Image& src, const Buffer<std::byte>& dst)
    : Command()
    , Source(std::cref(src))
    , Destination(std::cref(dst))
{}
RetrieveImageCommand::~RetrieveImageCommand() = default;

void UpdateImageCommand::record(vulkan::Command& cmd) const {
    //alias for shorter code
    auto& src = Source.get();
    auto& dst = Destination.get();
    //Check for src and dst to be from the same context
    if (src.getContext().get() != dst.getContext().get())
        throw std::logic_error(DIFFERENT_CONTEXT_ERROR_STR);
    auto& context = src.getContext();
    //check both have the same size
    if (src.size_bytes() != dst.size_bytes())
        throw std::logic_error(SIZE_MISMATCH_ERROR_STR);
    auto size = src.size_bytes();

    //we're acting on the transfer stage
    cmd.stage |= VK_PIPELINE_STAGE_TRANSFER_BIT;

    //make ensure image is safe to write and prepare it for the transfer
    VkImageMemoryBarrier barrier{
        .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask    = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask    = VK_ACCESS_TRANSFER_WRITE_BIT,
        .oldLayout        = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .image            = dst.getImage().image,
        .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
    };
    context->fnTable.vkCmdPipelineBarrier(cmd.buffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_DEPENDENCY_BY_REGION_BIT,
        0, nullptr,
        0, nullptr,
        1, &barrier);

    //issue copy
    VkBufferImageCopy copy{
        .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0 ,1 },
        .imageExtent = { dst.getWidth(), dst.getHeight(), dst.getDepth() }
    };
    context->fnTable.vkCmdCopyBufferToImage(cmd.buffer,
        src.getBuffer().buffer,
        dst.getImage().image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1, &copy);

    //make sure the image is ready and transfer to shader comp layout
    barrier = VkImageMemoryBarrier{
        .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask    = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask    = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        .oldLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .newLayout        = VK_IMAGE_LAYOUT_GENERAL,
        .image            = dst.getImage().image,
        .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
    };
    context->fnTable.vkCmdPipelineBarrier(cmd.buffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_DEPENDENCY_BY_REGION_BIT,
        0, nullptr,
        0, nullptr,
        1, &barrier);
}

UpdateImageCommand::UpdateImageCommand(const UpdateImageCommand& other) = default;
UpdateImageCommand& UpdateImageCommand::operator=(const UpdateImageCommand& other) = default;

UpdateImageCommand::UpdateImageCommand(UpdateImageCommand&& other) noexcept = default;
UpdateImageCommand& UpdateImageCommand::operator=(UpdateImageCommand&& other) noexcept = default;

UpdateImageCommand::UpdateImageCommand(const Buffer<std::byte>& src, const Image& dst)
    : Command()
    , Source(std::cref(src))
    , Destination(std::cref(dst))
{}
UpdateImageCommand::~UpdateImageCommand() = default;

void UpdateTextureCommand::record(vulkan::Command& cmd) const {
    //alias for shorter code
    auto& src = Source.get();
    auto& dst = Destination.get();
    //Check for src and dst to be from the same context
    if (src.getContext().get() != dst.getContext().get())
        throw std::logic_error(DIFFERENT_CONTEXT_ERROR_STR);
    auto& context = src.getContext();
    //check both have the same size
    if (src.size_bytes() != dst.size_bytes())
        throw std::logic_error(SIZE_MISMATCH_ERROR_STR);
    auto size = src.size_bytes();

    //we're acting on the transfer stage
    cmd.stage |= VK_PIPELINE_STAGE_TRANSFER_BIT;

    //make ensure image is safe to write and prepare it for the transfer
    VkImageMemoryBarrier barrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_READ_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .image = dst.getImage().image,
        .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
    };
    context->fnTable.vkCmdPipelineBarrier(cmd.buffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_DEPENDENCY_BY_REGION_BIT,
        0, nullptr,
        0, nullptr,
        1, &barrier);

    //issue copy
    VkBufferImageCopy copy{
        .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0 ,1 },
        .imageExtent = { dst.getWidth(), dst.getHeight(), dst.getDepth() }
    };
    context->fnTable.vkCmdCopyBufferToImage(cmd.buffer,
        src.getBuffer().buffer,
        dst.getImage().image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1, &copy);

    //make sure the image is ready and transfer to shader comp layout
    barrier = VkImageMemoryBarrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        .image = dst.getImage().image,
        .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
    };
    context->fnTable.vkCmdPipelineBarrier(cmd.buffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_DEPENDENCY_BY_REGION_BIT,
        0, nullptr,
        0, nullptr,
        1, &barrier);
}

UpdateTextureCommand::UpdateTextureCommand(const UpdateTextureCommand& other) = default;
UpdateTextureCommand& UpdateTextureCommand::operator=(const UpdateTextureCommand& other) = default;

UpdateTextureCommand::UpdateTextureCommand(UpdateTextureCommand&& other) noexcept = default;
UpdateTextureCommand& UpdateTextureCommand::operator=(UpdateTextureCommand&& other) noexcept = default;

UpdateTextureCommand::UpdateTextureCommand(const Buffer<std::byte>& src, const Texture& dst)
    : Command()
    , Source(std::cref(src))
    , Destination(std::cref(dst))
{}
UpdateTextureCommand::~UpdateTextureCommand() = default;

}
