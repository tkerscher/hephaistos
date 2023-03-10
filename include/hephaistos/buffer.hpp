#pragma once

#include <cstddef>
#include <initializer_list>
#include <span>

#include "hephaistos/config.hpp"
#include "hephaistos/context.hpp"
#include "hephaistos/handles.hpp"

namespace hephaistos {

template<class T = std::byte> class Buffer;

template<>
class Buffer<std::byte> : public Resource {
public:
	[[nodiscard]] std::span<std::byte> getMemory() const;
	[[nodiscard]] uint64_t size_bytes() const noexcept;
	[[nodiscard]] size_t size() const noexcept;

	Buffer(const Buffer<std::byte>&) = delete;
	Buffer<std::byte>& operator=(const Buffer<std::byte>&) = delete;

	Buffer(Buffer<std::byte>&& other) noexcept;
	Buffer<std::byte>& operator=(Buffer<std::byte>&& other) noexcept;

	Buffer(ContextHandle context, uint64_t size);
	Buffer(ContextHandle context, std::span<const std::byte> data);
	virtual ~Buffer();

public: //internal
	const vulkan::Buffer& getBuffer() const noexcept;

private:
	BufferHandle buffer;
	std::span<std::byte> memory;
};
template class HEPHAISTOS_API Buffer<std::byte>;

template<class T>
class Buffer : public Buffer<std::byte> {
public:
	[[nodiscard]] std::span<T> getMemory() const {
		auto mem = Buffer<std::byte>::getMemory();
		return {
			reinterpret_cast<T*>(mem.data()),
			mem.size() / sizeof(T)
		};
	}
	[[nodiscard]] size_t size() const noexcept {
		return size_bytes() / sizeof(T);
	}

	Buffer(const Buffer&) = delete;
	Buffer& operator=(const Buffer&) = delete;

	Buffer(Buffer&& other) noexcept
		: Buffer<std::byte>(std::move(other))
	{}
	Buffer& operator=(Buffer&& other) noexcept {
		Buffer<std::byte>::operator=(std::move(other));
		return *this;
	}

	Buffer(ContextHandle context, size_t count)
		: Buffer<std::byte>(std::move(context), count * sizeof(T))
	{}
	Buffer(ContextHandle context, std::span<const T> data)
		: Buffer<std::byte>(std::move(context), std::as_bytes(data))
	{}
	Buffer(ContextHandle context, std::initializer_list<T> data)
		: Buffer(std::move(context), std::span<const T>{data})
	{}
	virtual ~Buffer() = default;
};

template<class Container> Buffer(ContextHandle, const Container&)
	-> Buffer<typename Container::value_type>;

}
