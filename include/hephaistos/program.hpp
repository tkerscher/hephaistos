#pragma once

#include <optional>
#include <span>
#include <type_traits>
#include <vector>

#include "hephaistos/argument.hpp"
#include "hephaistos/buffer.hpp"
#include "hephaistos/command.hpp"
#include "hephaistos/imageformat.hpp"

namespace hephaistos {

namespace vulkan {
    struct Program;
}

struct SubgroupProperties {
    uint32_t subgroupSize;
    bool basicSupport;
    bool voteSupport;
    bool arithmeticSupport;
    bool ballotSupport;
    bool shuffleSupport;
    bool shuffleRelativeSupport;
    bool shuffleClusteredSupport;
    bool quadSupport;
};
[[nodiscard]] HEPHAISTOS_API SubgroupProperties
    getSubgroupProperties(const DeviceHandle& device);
[[nodiscard]] HEPHAISTOS_API SubgroupProperties
    getSubgroupProperties(const ContextHandle& context);

enum class ParameterType {
    /*SAMPLER = 0,*/
    COMBINED_IMAGE_SAMPLER = 1,
    //SAMPLED_IMAGE = 2,
    STORAGE_IMAGE = 3,
    //UNIFORM_TEXEL_BUFFER = 4,
    //STORAGE_TEXEL_BUFFER = 5,
    UNIFORM_BUFFER = 6,
    STORAGE_BUFFER = 7,
    //UNIFORM_BUFFER_DYNAMIC = 8,
    //STORAGE_BUFFER_DYNAMIC = 9,
    ACCELERATION_STRUCTURE = 1000150000
};

struct ImageBindingTraits {
    ImageFormat format;
    uint8_t dims;
};

struct BindingTraits {
    std::string name;
    uint32_t binding;
    ParameterType type;
    std::optional<ImageBindingTraits> imageTraits;
    uint32_t count; //0 for runtime array
};

class HEPHAISTOS_API DispatchCommand : public Command {
public:
    uint32_t groupCountX;
    uint32_t groupCountY;
    uint32_t groupCountZ;
    std::span<const std::byte> pushData;

    void record(vulkan::Command& cmd) const override;

    DispatchCommand(const DispatchCommand&);
    DispatchCommand& operator=(const DispatchCommand&);

    DispatchCommand(DispatchCommand&&) noexcept;
    DispatchCommand& operator=(DispatchCommand&&) noexcept;

    DispatchCommand(
        const vulkan::Program& program,
        uint32_t x, uint32_t y, uint32_t z,
        std::span<const std::byte> push);
    ~DispatchCommand() override;

private:
    std::reference_wrapper<const vulkan::Program> program;
    std::vector<VkWriteDescriptorSet> params;
};

struct DispatchIndirect {
    uint32_t groupCountX;
    uint32_t groupCountY;
    uint32_t groupCountZ;
};

class HEPHAISTOS_API DispatchIndirectCommand : public Command {
public:
    std::reference_wrapper<const Tensor<std::byte>> tensor;
    uint64_t offset;
    std::span<const std::byte> pushData;

    void record(vulkan::Command& cmd) const override;

    DispatchIndirectCommand(const DispatchIndirectCommand&);
    DispatchIndirectCommand& operator=(const DispatchIndirectCommand&);

    DispatchIndirectCommand(DispatchIndirectCommand&&) noexcept;
    DispatchIndirectCommand& operator=(DispatchIndirectCommand&&) noexcept;

    DispatchIndirectCommand(
        const vulkan::Program& program,
        const Tensor<std::byte>& tensor,
        uint64_t offset,
        std::span<const std::byte> push);
    ~DispatchIndirectCommand() override;

private:
    std::reference_wrapper<const vulkan::Program> program;
    std::vector<VkWriteDescriptorSet> params;
};

struct LocalSize {
    uint32_t x;
    uint32_t y;
    uint32_t z;
};

class HEPHAISTOS_API Program : public Resource {
public:
    [[nodiscard]] const LocalSize& getLocalSize() const noexcept;

    [[nodiscard]] const BindingTraits& getBindingTraits(uint32_t i) const;
    [[nodiscard]] const BindingTraits& getBindingTraits(std::string_view name) const;
    [[nodiscard]] const std::vector<BindingTraits>& listBindings() const noexcept;

    [[nodiscard]] VkWriteDescriptorSet& getBinding(uint32_t i);
    [[nodiscard]] VkWriteDescriptorSet& getBinding(std::string_view name);

    template<class T>
    void bindParameter(const T& param, uint32_t binding) {
        param.bindParameter(getBinding(binding));
    }
    template<class T>
    void bindParameter(const T& param, std::string_view binding) {
        param.bindParameter(getBinding(binding));
    }
    template<class ...T>
    void bindParameterList(const T&...param) {
        uint32_t binding = 0;
        (param.bindParameter(getBinding(binding++)),...);
    }

    [[nodiscard]] DispatchCommand dispatch(
        std::span<const std::byte> push,
        uint32_t x = 1, uint32_t y = 1, uint32_t z = 1) const;
    [[nodiscard]] DispatchCommand dispatch(
        uint32_t x = 1, uint32_t y = 1, uint32_t z = 1) const;
    template<class T, typename = typename std::enable_if_t<std::is_standard_layout_v<T> && !std::is_integral_v<T>>>
    [[nodiscard]] DispatchCommand dispatch(const T& push, uint32_t x = 1, uint32_t y = 1, uint32_t z = 1) const {
        return dispatch({ reinterpret_cast<const std::byte*>(&push), sizeof(T)}, x, y, z);
    }

    [[nodiscard]] DispatchIndirectCommand dispatchIndirect(
        std::span<const std::byte> push, const Tensor<std::byte>& tensor, uint64_t offset = 0) const;
    [[nodiscard]] DispatchIndirectCommand dispatchIndirect(
        const Tensor<std::byte>& tensor, uint64_t offset = 0) const;
    template<class T, typename = typename std::enable_if_t<std::is_standard_layout_v<T>>>
    [[nodiscard]] DispatchIndirectCommand dispatchIndirect(
        const T& push, const Tensor<std::byte>& tensor, uint64_t offset = 0) const
    {
        return dispatchIndirect({ reinterpret_cast<const std::byte*>(&push), sizeof(T) }, tensor, offset);
    }

    Program(const Program&) = delete;
    Program& operator=(const Program&) = delete;

    Program(Program&& other) noexcept;
    Program& operator=(Program&& other) noexcept;

    Program(ContextHandle context, std::span<const uint32_t> code);
    Program(ContextHandle context, std::span<const uint32_t> code, std::span<const std::byte> specialization);
    template<class T>
    Program(ContextHandle context, std::span<const uint32_t> code, const T& specialization)
        : Program(std::move(context), code, std::as_bytes(std::span<const T>{ &specialization, 1 }))
    {}
    ~Program() override;

private:
    std::unique_ptr<vulkan::Program> program;
    std::vector<BindingTraits> bindingTraits;
};

class HEPHAISTOS_API FlushMemoryCommand : public Command{
public:
    //The command on its own cant do anything so we're save to
    //just store the reference instead
    std::reference_wrapper<const vulkan::Context> context;

    void record(vulkan::Command& cmd) const override;

    FlushMemoryCommand(const FlushMemoryCommand&);
    FlushMemoryCommand& operator=(const FlushMemoryCommand&);

    explicit FlushMemoryCommand(const ContextHandle& context);
    ~FlushMemoryCommand() override;
};
[[nodiscard]] inline FlushMemoryCommand flushMemory(const ContextHandle& context) {
    return FlushMemoryCommand(context);
}

}
