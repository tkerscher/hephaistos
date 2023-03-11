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

template<class T = std::byte> class Tensor;

template<>
class Tensor<std::byte> : public Resource {
public:
    [[nodiscard]] uint64_t size_bytes() const noexcept;
    [[nodiscard]] size_t size() const noexcept;

    Tensor(const Tensor<std::byte>&) = delete;
    Tensor<std::byte>& operator=(const Tensor<std::byte>&) = delete;

    Tensor(Tensor<std::byte>&& other) noexcept;
    Tensor<std::byte>& operator=(Tensor<std::byte>&& other) noexcept;

    Tensor(ContextHandle context, uint64_t size);
    virtual ~Tensor();

public: //internal
    [[nodiscard]] const vulkan::Buffer& getBuffer() const noexcept;

private:
    BufferHandle buffer;
	uint64_t _size;
};
template class HEPHAISTOS_API Tensor<std::byte>;

template<class T>
class Tensor : public Tensor<std::byte> {
public:
    [[nodiscard]] uint64_t size() const noexcept {
        return Tensor<std::byte>::size_bytes() / sizeof(T);
    }

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor(Tensor&& other) noexcept
        : Tensor<std::byte>(std::move(other))
    {}
    Tensor& operator=(Tensor&& other) noexcept {
        Tensor<std::byte>::operator=(std::move(other));
        return *this;
    }

    Tensor(ContextHandle context, size_t count)
        : Tensor<std::byte>(std::move(context), count * sizeof(T))
    {}
    virtual ~Tensor() = default;
};

[[nodiscard]] HEPHAISTOS_API CommandHandle retrieveTensor(
	const Tensor<std::byte>& src, const Buffer<std::byte>& dst);
[[nodiscard]] HEPHAISTOS_API CommandHandle updateTensor(
	const Buffer<std::byte>& src, const Tensor<std::byte>& dst);

}
