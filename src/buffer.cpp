#include "hephaistos/buffer.hpp"

#include <algorithm>

#include "vk/types.hpp"
#include "vk/result.hpp"

namespace hephaistos {

/********************************** BUFFER ************************************/

std::span<std::byte> Buffer<std::byte>::getMemory() const {
	return memory;
}
uint64_t Buffer<std::byte>::size_bytes() const noexcept {
	return memory.size_bytes();
}
size_t Buffer<std::byte>::size() const noexcept {
	return memory.size();
}

const vulkan::Buffer& Buffer<std::byte>::getBuffer() const noexcept {
	return *buffer;
}

Buffer<std::byte>::Buffer(Buffer<std::byte>&& other) noexcept
	: Resource(std::move(other))
	, buffer(std::move(other.buffer))
{}
Buffer<std::byte>& Buffer<std::byte>::operator=(Buffer<std::byte>&& other) noexcept {
	Resource::operator=(std::move(other));
	buffer = std::move(other.buffer);
	return *this;
}

Buffer<std::byte>::Buffer(ContextHandle context, uint64_t size)
	: Resource(std::move(context))
	, buffer(vulkan::createBuffer(
		getContext(), size,
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT
	))
{
	memory = std::span<std::byte>(
		static_cast<std::byte*>(buffer->allocInfo.pMappedData),
		size);
}
Buffer<std::byte>::Buffer(ContextHandle context, std::span<const std::byte> data)
	: Buffer<std::byte>(std::move(context), data.size())
{
	//copy
	std::copy(data.begin(), data.end(), memory.begin());
}

Buffer<std::byte>::~Buffer() = default;

/********************************** TENSOR ************************************/

struct Tensor<std::byte>::Parameter {
	VkDescriptorBufferInfo buffer;
};

uint64_t Tensor<std::byte>::size_bytes() const noexcept {
	return _size;
}
size_t Tensor<std::byte>::size() const noexcept {
	return static_cast<size_t>(_size);
}
const vulkan::Buffer& Tensor<std::byte>::getBuffer() const noexcept {
	return *buffer;
}

void Tensor<std::byte>::bindParameter(VkWriteDescriptorSet& binding) const {
	binding.pNext            = nullptr;
	binding.pImageInfo       = nullptr;
	binding.pTexelBufferView = nullptr;
	binding.pBufferInfo      = &parameter->buffer;
}

Tensor<std::byte>::Tensor(Tensor<std::byte>&& other) noexcept
	: Resource(std::move(other))
	, buffer(std::move(other.buffer))
	, _size(other._size)
	, parameter(std::move(other.parameter))
{}
Tensor<std::byte>& Tensor<std::byte>::operator=(Tensor<std::byte>&& other) noexcept {
	Resource::operator=(std::move(other));
	buffer = std::move(other.buffer);
	_size = other._size;
	parameter = std::move(other.parameter);
	return *this;
}

Tensor<std::byte>::Tensor(ContextHandle context, uint64_t size)
	: Resource(std::move(context))
	, _size(size)
	, buffer(vulkan::createBuffer(
		getContext(), size,
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
		VK_BUFFER_USAGE_TRANSFER_DST_BIT |
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
		VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, //needed for ray tracing
		0))
	, parameter(std::make_unique<Parameter>())
{
	parameter->buffer = VkDescriptorBufferInfo{
		.buffer = buffer->buffer,
		.offset = 0,
		.range = VK_WHOLE_SIZE
	};
}
Tensor<std::byte>::~Tensor() = default;

/*********************************** COPY *************************************/

namespace {

constexpr auto DIFFERENT_CONTEXT_ERROR_STR =
	"Source and destination of a copy command must originate from the same context!";
constexpr auto SIZE_MISMATCH_ERROR_STR =
	"Source and destination must have the same size!";

}

void RetrieveTensorCommand::record(vulkan::Command& cmd) const {
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

	//ensure writing to tensor is finished
	VkBufferMemoryBarrier barrier{
		.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
		.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
		.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
		.buffer = src.getBuffer().buffer,
		.size = VK_WHOLE_SIZE
	};
	context->fnTable.vkCmdPipelineBarrier(cmd.buffer,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_DEPENDENCY_BY_REGION_BIT,
		0, nullptr,
		1, &barrier,
		0, nullptr);

	//actually copy the buffer
	VkBufferCopy copyRegion{
		.size = size
	};
	context->fnTable.vkCmdCopyBuffer(cmd.buffer,
		src.getBuffer().buffer,
		dst.getBuffer().buffer,
		1, &copyRegion);

	//barrier to ensure transfer finished
	barrier = VkBufferMemoryBarrier{
		.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
		.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
		.dstAccessMask = VK_ACCESS_HOST_READ_BIT,
		.buffer = dst.getBuffer().buffer,
		.size = VK_WHOLE_SIZE
	};
	context->fnTable.vkCmdPipelineBarrier(cmd.buffer,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_HOST_BIT,
		VK_DEPENDENCY_BY_REGION_BIT,
		0, nullptr,
		1, &barrier,
		0, nullptr);
}

RetrieveTensorCommand::RetrieveTensorCommand(const RetrieveTensorCommand& other) = default;
RetrieveTensorCommand& RetrieveTensorCommand::operator=(const RetrieveTensorCommand& other) = default;

RetrieveTensorCommand::RetrieveTensorCommand(RetrieveTensorCommand&& other) noexcept = default;
RetrieveTensorCommand& RetrieveTensorCommand::operator=(RetrieveTensorCommand&& other) noexcept = default;

RetrieveTensorCommand::RetrieveTensorCommand(const Tensor<std::byte>& src, const Buffer<std::byte>& dst)
	: Command()
	, Source(std::cref(src))
	, Destination(std::cref(dst))
{}
RetrieveTensorCommand::~RetrieveTensorCommand() = default;

void UpdateTensorCommand::record(vulkan::Command& cmd) const {
	//to shorten the code
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

	//ensure tensor is safe to update
	VkBufferMemoryBarrier barrier{
		.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
		.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
		.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
		.buffer = dst.getBuffer().buffer,
		.size = VK_WHOLE_SIZE
	};
	context->fnTable.vkCmdPipelineBarrier(cmd.buffer,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_DEPENDENCY_BY_REGION_BIT,
		0, nullptr,
		1, &barrier,
		0, nullptr);

	//actually copy the buffer
	VkBufferCopy copyRegion{
		.size = size
	};
	context->fnTable.vkCmdCopyBuffer(cmd.buffer,
		src.getBuffer().buffer,
		dst.getBuffer().buffer,
		1, &copyRegion);

	//barrier to ensure transfer finished
	barrier = VkBufferMemoryBarrier{
		.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
		.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
		.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
		.buffer = dst.getBuffer().buffer,
		.size = VK_WHOLE_SIZE
	};
	context->fnTable.vkCmdPipelineBarrier(cmd.buffer,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		VK_DEPENDENCY_BY_REGION_BIT,
		0, nullptr,
		1, &barrier,
		0, nullptr);
}

UpdateTensorCommand::UpdateTensorCommand(const UpdateTensorCommand& other) = default;
UpdateTensorCommand& UpdateTensorCommand::operator=(const UpdateTensorCommand& other) = default;

UpdateTensorCommand::UpdateTensorCommand(UpdateTensorCommand&& other) noexcept = default;
UpdateTensorCommand& UpdateTensorCommand::operator=(UpdateTensorCommand&& other) noexcept = default;

UpdateTensorCommand::UpdateTensorCommand(const Buffer<std::byte>& src, const Tensor<std::byte>& dst)
	: Command()
	, Source(std::cref(src))
	, Destination(std::cref(dst))
{}
UpdateTensorCommand::~UpdateTensorCommand() = default;

}
