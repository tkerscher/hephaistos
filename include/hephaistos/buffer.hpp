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

/**
 * @brief Allocates memory on the host for copying to and from the device
*/
template<>
class Buffer<std::byte> : public Resource {
public:
    /**
     * @brief Memory range the Buffer manages
    */
    [[nodiscard]] std::span<std::byte> getMemory() const;
    /**
     * @brief Size of the Buffer in bytes
    */
    [[nodiscard]] uint64_t size_bytes() const noexcept;
    /**
     * @brief Number of elements
    */
    [[nodiscard]] size_t size() const noexcept;

    Buffer(const Buffer<std::byte>&) = delete;
    Buffer<std::byte>& operator=(const Buffer<std::byte>&) = delete;

    Buffer(Buffer<std::byte>&& other) noexcept;
    Buffer<std::byte>& operator=(Buffer<std::byte>&& other) noexcept;

    /**
     * @brief Allocates a new buffer
     * 
     * @param context Context onto which to create the buffer
     * @param size Number of elements
    */
    Buffer(ContextHandle context, uint64_t size);
    /**
     * @brief Allocates a new buffer
     * 
     * @param context Context onto which to create the buffer
     * @param data Data to fill the Buffer with
    */
    Buffer(ContextHandle context, std::span<const std::byte> data);
    ~Buffer() override;

public: //internal
    const vulkan::Buffer& getBuffer() const noexcept;

private:
    BufferHandle buffer;
    std::span<std::byte> memory;
};
template class HEPHAISTOS_API Buffer<std::byte>;

/**
 * @brief Allocates memory on the host for copying to and from the device
*/
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

/**
 * @brief Allocates memory on the device
*/
template<>
class Tensor<std::byte> : public Argument, public Resource {
public:
    /**
     * @brief Returns device memory address
    */
    [[nodiscard]] uint64_t address() const noexcept;
    /**
     * @brief Returns the tensor memory mapped to host memory space
     * 
     * @note Mapping memory may not supported by the device, in which case the
     *       returned range is empty.
    */
    [[nodiscard]] std::span<std::byte> getMemory() const;

    /**
     * @brief Wether the tensor is mapped to host memory space
     * 
     * @note Mapping memory may not supported by the device, in which case this
     *       will be false even if requested.
    */
    [[nodiscard]] bool isMapped() const noexcept;
    /**
     * @brief Size of the tensor in bytes
    */
    [[nodiscard]] uint64_t size_bytes() const noexcept;
    /**
     * @brief Number of elements
    */
    [[nodiscard]] size_t size() const noexcept;

    void bindParameter(VkWriteDescriptorSet& binding) const final override;

    Tensor(const Tensor<std::byte>&) = delete;
    Tensor<std::byte>& operator=(const Tensor<std::byte>&) = delete;

    Tensor(Tensor<std::byte>&& other) noexcept;
    Tensor<std::byte>& operator=(Tensor<std::byte>&& other) noexcept;

    /**
     * @brief Allocates a new Tensor
     * 
     * @param context Context onto which to create the Tensor
     * @param size Number of elements
     * @param mapped If true, tries to map the allocated tensor to host memory space
     * 
     * @note Mapping may not supported by the device. Check for success via
     *       isMapped().
    */
    Tensor(ContextHandle context, uint64_t size, bool mapped = false);
    /**
     * @brief Allocates a new Tensor
     * 
     * @param source Source buffer to copy from
     * @param mapped If true, tries to map the allocated tensor to host memory space
     * 
     * @note Mapping may not supported by the device. Check for success via
     *       isMapped().
    */
    explicit Tensor(const Buffer<std::byte>& source, bool mapped = false);
    /**
     * @brief Allocates a new Tensor
     * 
     * @param context Context onto which to create the Tensor
     * @param data Data the tensor will be initialized with
     * @param mapped If true, tries to map the allocated tensor to host memory space
     * 
     * @note Mapping may not supported by the device. Check for success via
     *       isMapped().
    */
    Tensor(ContextHandle context, std::span<const std::byte> data, bool mapped = false);
    ~Tensor() override;

public: //internal
    [[nodiscard]] const vulkan::Buffer& getBuffer() const noexcept;

private:
    uint64_t _size;
    BufferHandle buffer;

    struct Parameter;
    std::unique_ptr<Parameter> parameter;
};
template class HEPHAISTOS_API Tensor<std::byte>;

template<class T>
class Tensor : public Tensor<std::byte> {
public:
    [[nodiscard]] std::span<T> getMemory() const {
        if (!isMapped()) return {};

        auto mem = Tensor<std::byte>::getMemory();
        return {
            reinterpret_cast<T*>(mem.data()),
            mem.size() / sizeof(T)
        };
    }

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

    Tensor(ContextHandle context, size_t count, bool mapped = false)
        : Tensor<std::byte>(std::move(context), count * sizeof(T), mapped)
    {}
    explicit Tensor(const Buffer<std::byte>& buffer, bool mapped = false)
        : Tensor<std::byte>(buffer, mapped)
    {}
    Tensor(ContextHandle context, std::span<const T> data, bool mapped = false)
        : Tensor<std::byte>(std::move(context), std::as_bytes(data), mapped)
    {}
    Tensor(ContextHandle context, const T& data, bool mapped = false) requires (!std::is_integral_v<T>)
        : Tensor<std::byte>(
            std::move(context),
            { reinterpret_cast<const std::byte*>(&data), sizeof(T) },
            mapped)
    {}
    ~Tensor() override = default;
};

/**
 * @brief Magic number indicating complete memory size
*/
constexpr uint64_t whole_size = (~0ULL);
/**
 * @brief Description of memory region for copying
*/
struct CopyRegion {
    /**
     * @brief Offset into the Buffer
    */
    uint64_t bufferOffset = 0;
    /**
     * @brief Offset into the Tensor
    */
    uint64_t tensorOffset = 0;
    /**
     * @brief Number of bytes to copy
    */
    uint64_t size = whole_size;
};

/**
 * @brief Command for copying data from a Tensor to a Buffer
*/
class HEPHAISTOS_API RetrieveTensorCommand : public Command {
public:
    /**
     * @brief Source Tensor to copy from
    */
    std::reference_wrapper<const Tensor<std::byte>> source;
    /**
     * @brief Destination Buffer to copy to
    */
    std::reference_wrapper<const Buffer<std::byte>> destination;

    /**
     * @brief Offset into the source Tensor in bytes
    */
    uint64_t sourceOffset;
    /**
     * @brief Offset into the destination Buffer in bytes
    */
    uint64_t destinationOffset;
    /**
     * @brief Number of bytes to copy
    */
    uint64_t size;

    virtual void record(vulkan::Command& cmd) const override;

    RetrieveTensorCommand(const RetrieveTensorCommand& other);
    RetrieveTensorCommand& operator=(const RetrieveTensorCommand& other);

    RetrieveTensorCommand(RetrieveTensorCommand&& other) noexcept;
    RetrieveTensorCommand& operator=(RetrieveTensorCommand&& other) noexcept;
    
    /**
     * @brief Creates a new RetrieveTensorCommand
     * 
     * @param src Source Tensor to copy from
     * @param dst Destination Buffer to copy to
     * @param region Region to copy
    */
    RetrieveTensorCommand(
        const Tensor<std::byte>& src,
        const Buffer<std::byte>& dst,
        const CopyRegion& region = {});
    ~RetrieveTensorCommand() override;
};
/**
 * @brief Creates a new RetrieveTensorCommand
 * 
 * @param src Source Tensor to copy from
 * @param dst Destination Buffer to copy to
 * @param region Region to copy
*/
[[nodiscard]] inline RetrieveTensorCommand retrieveTensor(
    const Tensor<std::byte>& src,
    const Buffer<std::byte>& dst,
    const CopyRegion& region = {})
{
    return RetrieveTensorCommand(src, dst, region);
}

/**
 * @brief Command for copying data from Buffer to Tensor
*/
class HEPHAISTOS_API UpdateTensorCommand : public Command {
public:
    /**
     * @brief Source Buffer to copy from
    */
    std::reference_wrapper<const Buffer<std::byte>> source;
    /**
     * @brief Destination Tensor to copy to
    */
    std::reference_wrapper<const Tensor<std::byte>> destination;

    /**
     * @brief Offset into the source Buffer in bytes
    */
    uint64_t sourceOffset;
    /**
     * @brief Offset into the destination Tensor in bytes
    */
    uint64_t destinationOffset;
    /**
     * @brief Number of bytes to copy
    */
    uint64_t size;

    virtual void record(vulkan::Command& cmd) const override;

    UpdateTensorCommand(const UpdateTensorCommand& other);
    UpdateTensorCommand& operator=(const UpdateTensorCommand& other);

    UpdateTensorCommand(UpdateTensorCommand&& other) noexcept;
    UpdateTensorCommand& operator=(UpdateTensorCommand&& other) noexcept;

    /**
     * @brief Creates a new UpdateTensorCommand
     * 
     * @param src Source Buffer to copy from
     * @param dst Destination Tensor to copy to
     * @param region Region to copy
    */
    UpdateTensorCommand(
        const Buffer<std::byte>& src,
        const Tensor<std::byte>& dst,
        const CopyRegion& region = {});
    ~UpdateTensorCommand() override;
};
/**
 * @brief Creates a UpdateTensorCommand
 * 
 * @param src Source Buffer to copy from
 * @param dst Destination Tensor to copy to
 * @param region Region to copy
*/
[[nodiscard]] inline UpdateTensorCommand updateTensor(
    const Buffer<std::byte>& src,
    const Tensor<std::byte>& dst,
    const CopyRegion& region = {})
{
    return UpdateTensorCommand(src, dst, region);
}

/**
 * @brief Creates a ClearTensorCommand
 * 
 * Fills the given tensor with a constant 4 byte value. Amount and offset can
 * be specified.
 * 
 * @note Offset and size must be multiple of 4
*/
class HEPHAISTOS_API ClearTensorCommand : public Command {
public:
    struct Params {
        uint64_t offset = 0;
        uint64_t size   = whole_size;
        uint32_t data   = 0;
    };

public:
    /**
     * @brief Target Tensor to clear
    */
    std::reference_wrapper<const Tensor<std::byte>> tensor;
    /**
     * @brief Offset into the target Tensor in bytes
     * 
     * @note Must be a multiple of 4
    */
    uint64_t offset;
    /**
     * @brief Size of the region to clear in bytes
     * 
     * @note Must be a multiple of 4. whole_size, however, is always truncated
     *       to the remaining rest of the Tensor.
    */
    uint64_t size;
    /**
     * @brief Value used to clear the Tensor
    */
    uint32_t data;

    virtual void record(vulkan::Command& cmd) const override;

    ClearTensorCommand(const ClearTensorCommand& other);
    ClearTensorCommand& operator=(const ClearTensorCommand& other);

    ClearTensorCommand(ClearTensorCommand&& other) noexcept;
    ClearTensorCommand& operator=(ClearTensorCommand&& other) noexcept;

    /**
     * @brief Creates a new ClearTensorCommand
     * 
     * @param tensor Target Tensor to clear
     * @param params Parametrization of the clear command
     * 
     * @note Offset and size must be a multiple of 4
    */
    ClearTensorCommand(const Tensor<std::byte>& tensor, const Params& params);
    ~ClearTensorCommand() override;
};
/**
 * @brief Creates a ClearTensorCommand
 * 
 * @param tensor Target Tensor to clear
 * @param params Parameterization of the clear command
 * 
 * @note Offset and size must be a multiple of 4
*/
[[nodiscard]] inline ClearTensorCommand clearTensor(
    const Tensor<std::byte>& tensor, const ClearTensorCommand::Params& params)
{
    return ClearTensorCommand(tensor, params);
}

}
