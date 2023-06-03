#pragma once

#include <cstddef>
#include <initializer_list>
#include <span>

#include "hephaistos/argument.hpp"
#include "hephaistos/command.hpp"
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
	~Buffer() override;

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
	~Buffer() override = default;
};

template<class Container> Buffer(ContextHandle, const Container&)
	-> Buffer<typename Container::value_type>;

template<class T = std::byte> class Tensor;

template<>
class Tensor<std::byte> : public Argument, public Resource {
public:
    [[nodiscard]] uint64_t size_bytes() const noexcept;
    [[nodiscard]] size_t size() const noexcept;

	void bindParameter(VkWriteDescriptorSet& binding) const override final;

    Tensor(const Tensor<std::byte>&) = delete;
    Tensor<std::byte>& operator=(const Tensor<std::byte>&) = delete;

    Tensor(Tensor<std::byte>&& other) noexcept;
    Tensor<std::byte>& operator=(Tensor<std::byte>&& other) noexcept;

    Tensor(ContextHandle context, uint64_t size);
	Tensor(const Buffer<std::byte>& source);
	Tensor(ContextHandle context, std::span<const std::byte> data);
    ~Tensor() override;

public: //internal
    [[nodiscard]] const vulkan::Buffer& getBuffer() const noexcept;

private:
	uint64_t _size;
    BufferHandle buffer;

	struct Parameter;
	std::unique_ptr<Parameter> parameter;

	friend class ArgumentArray<Tensor<std::byte>>;
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
	Tensor(const Buffer<std::byte>& buffer)
		: Tensor<std::byte>(buffer)
	{}
	Tensor(ContextHandle context, std::span<const T> data)
		: Tensor<std::byte>(std::move(context), std::as_bytes(data))
	{}
	template<typename = std::enable_if_t<!std::is_integral_v<T>>>
	Tensor(ContextHandle context, const T& data)
		: Tensor<std::byte>(std::move(context), { reinterpret_cast<const std::byte*>(&data), sizeof(T) })
	{}
    virtual ~Tensor() = default;
};

template<>
class HEPHAISTOS_API ArgumentArray<Tensor<std::byte>> : public Argument {
public:
	using ElementType = Tensor<std::byte>;

public:
	void bindParameter(VkWriteDescriptorSet& binding) const override final;

	[[nodiscard]] size_t size() const;
	[[nodiscard]] bool empty() const;

	void clear();
	void resize(size_t count);

	void set(size_t pos, const ElementType& tensor);
	void push_back(const ElementType& tensor);

	ArgumentArray(const ArgumentArray& other);
	ArgumentArray& operator=(const ArgumentArray& other);

	ArgumentArray(ArgumentArray&& other) noexcept;
	ArgumentArray& operator=(ArgumentArray&& other) noexcept;

	ArgumentArray();
	ArgumentArray(size_t count);
	ArgumentArray(size_t count, const ElementType& tensor);
	ArgumentArray(std::initializer_list<std::reference_wrapper<const ElementType>> tensors);

	~ArgumentArray() override;

private:
	struct Parameter;
	std::unique_ptr<Parameter> parameter;
};

class HEPHAISTOS_API RetrieveTensorCommand : public Command {
public:
	std::reference_wrapper<const Tensor<std::byte>> Source;
	std::reference_wrapper<const Buffer<std::byte>> Destination;

	void record(vulkan::Command& cmd) const override;

	RetrieveTensorCommand(const RetrieveTensorCommand& other);
	RetrieveTensorCommand& operator=(const RetrieveTensorCommand& other);

	RetrieveTensorCommand(RetrieveTensorCommand&& other) noexcept;
	RetrieveTensorCommand& operator=(RetrieveTensorCommand&& other) noexcept;
	
	RetrieveTensorCommand(const Tensor<std::byte>& src, const Buffer<std::byte>& dst);
	~RetrieveTensorCommand() override;
};
[[nodiscard]] inline RetrieveTensorCommand retrieveTensor(
	const Tensor<std::byte>& src, const Buffer<std::byte>& dst)
{
	return RetrieveTensorCommand(src, dst);
}

class HEPHAISTOS_API UpdateTensorCommand : public Command {
public:
	std::reference_wrapper<const Buffer<std::byte>> Source;
	std::reference_wrapper<const Tensor<std::byte>> Destination;

	void record(vulkan::Command& cmd) const override;

	UpdateTensorCommand(const UpdateTensorCommand& other);
	UpdateTensorCommand& operator=(const UpdateTensorCommand& other);

	UpdateTensorCommand(UpdateTensorCommand&& other) noexcept;
	UpdateTensorCommand& operator=(UpdateTensorCommand&& other) noexcept;

	UpdateTensorCommand(const Buffer<std::byte>& src, const Tensor<std::byte>& dst);
	~UpdateTensorCommand() override;
};
[[nodiscard]] inline UpdateTensorCommand updateTensor(
	const Buffer<std::byte>& src, const Tensor<std::byte>& dst)
{
	return UpdateTensorCommand(src, dst);
}

}
