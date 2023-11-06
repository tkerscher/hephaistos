#include "vk/types.hpp"

#include "vk/result.hpp"
#include "vk/util.hpp"

namespace hephaistos::vulkan {

BufferHandle createBuffer(
    const ContextHandle& context,
    uint64_t size,
    VkBufferUsageFlags usage,
    VmaAllocationCreateFlags flags)
{
    BufferHandle result{ new Buffer({0,0,{},*context}), destroyBuffer };

    VkBufferCreateInfo bufferInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
    };
    VmaAllocationCreateInfo allocInfo{
        .flags = flags,
        .usage = VMA_MEMORY_USAGE_AUTO
    };

    checkResult(vmaCreateBuffer(
        context->allocator,
        &bufferInfo,
        &allocInfo,
        &result->buffer,
        &result->allocation,
        &result->allocInfo));

    return result;
}
void destroyBuffer(Buffer* buffer) {
    if (!buffer)
        return;
    vmaDestroyBuffer(buffer->context.allocator, buffer->buffer, buffer->allocation);
}

ImageHandle createImage(
    const ContextHandle& context,
    VkFormat format,
    uint32_t width, uint32_t height, uint32_t depth,
    VkImageUsageFlags usage)
{
    ImageHandle result{ new Image({ 0, 0, {}, *context}), destroyImage };

    //Determine dimension of image
    VkImageType imageType = VK_IMAGE_TYPE_3D;
    VkImageViewType viewType = VK_IMAGE_VIEW_TYPE_3D;
    if (depth == 1 && height == 1) {
        imageType = VK_IMAGE_TYPE_1D;
        viewType = VK_IMAGE_VIEW_TYPE_1D;
    }
    else if (depth == 1) {
        imageType = VK_IMAGE_TYPE_2D;
        viewType = VK_IMAGE_VIEW_TYPE_2D;
    }

    //Create image
    VkImageCreateInfo imageInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = imageType,
        .format = format,
        .extent = { width, height, depth },
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = usage
    };
    VmaAllocationCreateInfo allocInfo{
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
    };
    checkResult(vmaCreateImage(
        context->allocator, &imageInfo, &allocInfo,
        &result->image, &result->allocation, nullptr));

    //create view
    VkImageViewCreateInfo viewInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = result->image,
        .viewType = viewType,
        .format = format,
        .subresourceRange = {
            VK_IMAGE_ASPECT_COLOR_BIT,
            0, 1, 0, 1
        }
    };
    checkResult(context->fnTable.vkCreateImageView(
        context->device, &viewInfo, nullptr, &result->view));

    //done
    return result;
}
void destroyImage(Image* image) {
    if (!image)
        return;

    image->context.fnTable.vkDestroyImageView(
        image->context.device, image->view, nullptr);
    vmaDestroyImage(image->context.allocator,
        image->image, image->allocation);
}

}
