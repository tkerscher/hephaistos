#include "hephaistos/buffer.hpp"

#include <algorithm>
#include <array>

#include "vk/util.hpp"
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
    uint64_t address;
    VkDescriptorBufferInfo buffer;
};

uint64_t Tensor<std::byte>::address() const noexcept {
    return parameter->address;
}

std::span<std::byte> Tensor<std::byte>::getMemory() const {
    if (!isMapped()) return {};

    return { static_cast<std::byte*>(buffer->allocInfo.pMappedData), _size };
}

bool Tensor<std::byte>::isMapped() const noexcept {
    return buffer->allocInfo.pMappedData != nullptr;
}
bool Tensor<std::byte>::isNonCoherent() const noexcept {
    VkMemoryPropertyFlags flags;
    vmaGetAllocationMemoryProperties(
        getContext()->allocator,
        buffer->allocation,
        &flags);
    return (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0;
}
void Tensor<std::byte>::update(std::span<const std::byte> src, uint64_t offset) {
    vulkan::checkResult(vmaCopyMemoryToAllocation(
        getContext()->allocator,
        static_cast<const void*>(src.data()),
        buffer->allocation,
        offset,
        src.size_bytes()
    ));
}
void Tensor<std::byte>::flush(uint64_t offset, uint64_t size) {
    vulkan::checkResult(vmaFlushAllocation(
        getContext()->allocator,
        buffer->allocation,
        offset, size
    ));
}
void Tensor<std::byte>::retrieve(std::span<std::byte> dst, uint64_t offset) {
    vulkan::checkResult(vmaCopyAllocationToMemory(
        getContext()->allocator,
        buffer->allocation,
        offset,
        static_cast<void*>(dst.data()),
        dst.size_bytes()
    ));
}
void Tensor<std::byte>::invalidate(uint64_t offset, uint64_t size) {
    vulkan::checkResult(vmaInvalidateAllocation(
        getContext()->allocator,
        buffer->allocation,
        offset, size
    ));
}

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

namespace {

constexpr VkBufferUsageFlags tensor_usage =
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
    VK_BUFFER_USAGE_TRANSFER_DST_BIT |
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
    VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

constexpr VmaAllocationCreateFlags tensor_mapped_flags =
    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
    VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT |
    VMA_ALLOCATION_CREATE_MAPPED_BIT;

}

Tensor<std::byte>::Tensor(ContextHandle context, uint64_t size, bool mapped)
    : Resource(std::move(context))
    , _size(size)
    , buffer(vulkan::createBuffer(
        getContext(), size,
        tensor_usage,
        mapped ? tensor_mapped_flags : 0))
    , parameter(std::make_unique<Parameter>())
{
    parameter->buffer = VkDescriptorBufferInfo{
        .buffer = buffer->buffer,
        .offset = 0,
        .range = VK_WHOLE_SIZE
    };

    VkBufferDeviceAddressInfo addressInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .buffer = buffer->buffer
    };
    parameter->address = getContext()->fnTable.vkGetBufferDeviceAddress(
        getContext()->device, &addressInfo);
}
Tensor<std::byte>::Tensor(const Buffer<std::byte>& source, bool mapped)
    : Tensor<std::byte>(source.getContext(), source.size_bytes(), mapped)
{
    //one time submit copy buffer to source
    UpdateTensorCommand command(source, *this);
    vulkan::oneTimeSubmit(*getContext(), [&command](VkCommandBuffer cmd) {
        vulkan::Command wrapper{ cmd, 0 };
        command.record(wrapper);
    });
}
Tensor<std::byte>::Tensor(ContextHandle context, std::span<const std::byte> data, bool mapped)
    : Tensor<std::byte>(Buffer<std::byte>(std::move(context), data), mapped)
{}
Tensor<std::byte>::~Tensor() = default;

/*********************************** COMMANDS *************************************/

namespace {

constexpr auto DIFFERENT_CONTEXT_ERROR_STR =
    "Source and destination of a copy command must originate from the same context!";
constexpr auto SIZE_MISMATCH_ERROR_STR =
    "Source and destination copy region must have the same size!";
constexpr auto COPY_REGION_OUT_OF_SOURCE =
    "Copy region is not contained within the source!";
constexpr auto COPY_REGION_OUT_OF_DESTINATION =
    "Copy region is not contained within the destination!";

}

void RetrieveTensorCommand::record(vulkan::Command& cmd) const {
    //alias for shorter code
    auto& src = source.get();
    auto& dst = destination.get();
    //Check for src and dst to be from the same context
    if (src.getContext().get() != dst.getContext().get())
        throw std::logic_error(DIFFERENT_CONTEXT_ERROR_STR);
    auto& context = src.getContext();
    //check both have the same size
    auto srcSize = (size == whole_size ? src.size_bytes() - sourceOffset : size);
    auto dstSize = (size == whole_size ? dst.size_bytes() - destinationOffset : size);
    if (srcSize != dstSize)
        throw std::logic_error(SIZE_MISMATCH_ERROR_STR);
    //shadow size with actual value
    auto size = srcSize;
    //boundary check
    if (size + sourceOffset > src.size_bytes())
        throw std::logic_error(COPY_REGION_OUT_OF_SOURCE);
    if (size + destinationOffset > dst.size_bytes())
        throw std::logic_error(COPY_REGION_OUT_OF_DESTINATION);

    //we're acting on the transfer stage
    cmd.stage |= VK_PIPELINE_STAGE_TRANSFER_BIT;

    //ensure writing to tensor is finished
    if (!unsafe) {
        std::array<VkBufferMemoryBarrier, 2> barriers{
            VkBufferMemoryBarrier{
                .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
                .buffer = src.getBuffer().buffer,
                .offset = sourceOffset,
                .size = size
            },
            VkBufferMemoryBarrier{
                .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .buffer = dst.getBuffer().buffer,
                .offset = destinationOffset,
                .size = size
            }
        };
        context->fnTable.vkCmdPipelineBarrier(cmd.buffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_DEPENDENCY_BY_REGION_BIT,
            0, nullptr,
            static_cast<uint32_t>(barriers.size()), barriers.data(),
            0, nullptr);
    }

    //actually copy the buffer
    VkBufferCopy copyRegion{
        .srcOffset = sourceOffset,
        .dstOffset = destinationOffset,
        .size = size
    };
    context->fnTable.vkCmdCopyBuffer(cmd.buffer,
        src.getBuffer().buffer,
        dst.getBuffer().buffer,
        1, &copyRegion);

    //barrier to ensure transfer finished
    if (!unsafe) {
        VkBufferMemoryBarrier barrier{
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
            .buffer = dst.getBuffer().buffer,
            .offset = destinationOffset,
            .size = size
        };
        context->fnTable.vkCmdPipelineBarrier(cmd.buffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_HOST_BIT,
            VK_DEPENDENCY_BY_REGION_BIT,
            0, nullptr,
            1, &barrier,
            0, nullptr);
    }
}

RetrieveTensorCommand::RetrieveTensorCommand(const RetrieveTensorCommand& other) = default;
RetrieveTensorCommand& RetrieveTensorCommand::operator=(const RetrieveTensorCommand& other) = default;

RetrieveTensorCommand::RetrieveTensorCommand(RetrieveTensorCommand&& other) noexcept = default;
RetrieveTensorCommand& RetrieveTensorCommand::operator=(RetrieveTensorCommand&& other) noexcept = default;

RetrieveTensorCommand::RetrieveTensorCommand(
    const Tensor<std::byte>& src,
    const Buffer<std::byte>& dst,
    const CopyRegion& region)
    : Command()
    , source(std::cref(src))
    , destination(std::cref(dst))
    , sourceOffset(region.tensorOffset)
    , destinationOffset(region.bufferOffset)
    , size(region.size)
    , unsafe(region.unsafe)
{}
RetrieveTensorCommand::~RetrieveTensorCommand() = default;

void UpdateTensorCommand::record(vulkan::Command& cmd) const {
    //to shorten the code
    auto& src = source.get();
    auto& dst = destination.get();
    //Check for src and dst to be from the same context
    if (src.getContext().get() != dst.getContext().get())
        throw std::logic_error(DIFFERENT_CONTEXT_ERROR_STR);
    auto& context = src.getContext();
    //check both have the same size
    auto srcSize = (size == whole_size ? src.size_bytes() - sourceOffset : size);
    auto dstSize = (size == whole_size ? dst.size_bytes() - destinationOffset : size);
    if (srcSize != dstSize)
        throw std::logic_error(SIZE_MISMATCH_ERROR_STR);
    //shadow size with actual size
    auto size = srcSize;
    //boundary check
    if (size + sourceOffset > src.size_bytes())
        throw std::logic_error(COPY_REGION_OUT_OF_SOURCE);
    if (size + destinationOffset > dst.size_bytes())
        throw std::logic_error(COPY_REGION_OUT_OF_DESTINATION);

    //we're acting on the transfer stage
    cmd.stage |= VK_PIPELINE_STAGE_TRANSFER_BIT;

    //ensure tensor is safe to update
    if (!unsafe) {
        VkBufferMemoryBarrier barrier{
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .buffer = dst.getBuffer().buffer,
            .offset = destinationOffset,
            .size = size
        };
        context->fnTable.vkCmdPipelineBarrier(cmd.buffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_DEPENDENCY_BY_REGION_BIT,
            0, nullptr,
            1, &barrier,
            0, nullptr);
        barrier = VkBufferMemoryBarrier{
                .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_HOST_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
                .buffer = src.getBuffer().buffer,
                .offset = sourceOffset,
                .size = size
        };
        context->fnTable.vkCmdPipelineBarrier(cmd.buffer,
            VK_PIPELINE_STAGE_HOST_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_DEPENDENCY_BY_REGION_BIT,
            0, nullptr,
            1, &barrier,
            0, nullptr);
    }

    //actually copy the buffer
    VkBufferCopy copyRegion{
        .srcOffset = sourceOffset,
        .dstOffset = destinationOffset,
        .size = size
    };
    context->fnTable.vkCmdCopyBuffer(cmd.buffer,
        src.getBuffer().buffer,
        dst.getBuffer().buffer,
        1, &copyRegion);

    //barrier to ensure transfer finished
    if (!unsafe) {
        VkBufferMemoryBarrier barrier{
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
            .buffer = dst.getBuffer().buffer,
            .offset = destinationOffset,
            .size = size
        };
        context->fnTable.vkCmdPipelineBarrier(cmd.buffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_DEPENDENCY_BY_REGION_BIT,
            0, nullptr,
            1, &barrier,
            0, nullptr);
    }
}

UpdateTensorCommand::UpdateTensorCommand(const UpdateTensorCommand& other) = default;
UpdateTensorCommand& UpdateTensorCommand::operator=(const UpdateTensorCommand& other) = default;

UpdateTensorCommand::UpdateTensorCommand(UpdateTensorCommand&& other) noexcept = default;
UpdateTensorCommand& UpdateTensorCommand::operator=(UpdateTensorCommand&& other) noexcept = default;

UpdateTensorCommand::UpdateTensorCommand(
    const Buffer<std::byte>& src,
    const Tensor<std::byte>& dst,
    const CopyRegion& region)
    : Command()
    , source(std::cref(src))
    , destination(std::cref(dst))
    , sourceOffset(region.bufferOffset)
    , destinationOffset(region.tensorOffset)
    , size(region.size)
    , unsafe(region.unsafe)
{}
UpdateTensorCommand::~UpdateTensorCommand() = default;


void ClearTensorCommand::record(vulkan::Command& cmd) const {
    //to shorten things
    auto& context = tensor.get().getContext();
    auto buffer = tensor.get().getBuffer().buffer; //ptr -> by value

    //we're acting on the transfer stage
    cmd.stage |= VK_PIPELINE_STAGE_TRANSFER_BIT;

    //ensure tensor is safe to update
    if (!unsafe) {
        VkBufferMemoryBarrier barrier{
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .buffer = buffer,
            .size = VK_WHOLE_SIZE
        };
        context->fnTable.vkCmdPipelineBarrier(cmd.buffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_DEPENDENCY_BY_REGION_BIT,
            0, nullptr,
            1, &barrier,
            0, nullptr);
    }

    //fill buffer
    context->fnTable.vkCmdFillBuffer(cmd.buffer,
        buffer, offset, size, data);

    //barrier to ensure transfer finished
    if (!unsafe) {
        VkBufferMemoryBarrier barrier{
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
            .buffer = buffer,
            .size = VK_WHOLE_SIZE
        };
        context->fnTable.vkCmdPipelineBarrier(cmd.buffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_DEPENDENCY_BY_REGION_BIT,
            0, nullptr,
            1, &barrier,
            0, nullptr);
    }
}

ClearTensorCommand::ClearTensorCommand(const ClearTensorCommand& other) = default;
ClearTensorCommand& ClearTensorCommand::operator=(const ClearTensorCommand& other) = default;

ClearTensorCommand::ClearTensorCommand(ClearTensorCommand&& other) noexcept = default;
ClearTensorCommand& ClearTensorCommand::operator=(ClearTensorCommand&& other) noexcept = default;

ClearTensorCommand::ClearTensorCommand(const Tensor<std::byte>& tensor, const Params& params)
    : Command()
    , tensor(std::cref(tensor))
    , offset(params.offset)
    , size(params.size)
    , data(params.data)
    , unsafe(params.unsafe)
{}
ClearTensorCommand::~ClearTensorCommand() = default;

}
