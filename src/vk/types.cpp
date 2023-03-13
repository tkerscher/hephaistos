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
	uint32_t width, uint32_t height, uint32_t depth)
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
		.usage = VK_IMAGE_USAGE_STORAGE_BIT |
			VK_IMAGE_USAGE_TRANSFER_DST_BIT |
			VK_IMAGE_USAGE_TRANSFER_SRC_BIT
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

	//transition image from undefined to general
	VkImage image = result->image;
	oneTimeSubmit(*context, [image, &context](VkCommandBuffer cmdBuffer) {
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
		context->fnTable.vkCmdPipelineBarrier(cmdBuffer,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			VK_DEPENDENCY_BY_REGION_BIT,
			0, nullptr,
			0, nullptr,
			1, &barrier);
	});

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
