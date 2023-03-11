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

uint64_t Tensor<std::byte>::size_bytes() const noexcept {
	return _size;
}
size_t Tensor<std::byte>::size() const noexcept {
	return static_cast<size_t>(_size);
}
const vulkan::Buffer& Tensor<std::byte>::getBuffer() const noexcept {
	return *buffer;
}

Tensor<std::byte>::Tensor(Tensor<std::byte>&& other) noexcept
	: Resource(std::move(other))
	, buffer(std::move(other.buffer))
	, _size(other._size)
{}
Tensor<std::byte>& Tensor<std::byte>::operator=(Tensor<std::byte>&& other) noexcept {
	Resource::operator=(std::move(other));
	buffer = std::move(other.buffer);
	_size = other._size;
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
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		0
	))
{}
Tensor<std::byte>::~Tensor() = default;

}
