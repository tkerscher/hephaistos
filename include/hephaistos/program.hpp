#pragma once

#include <optional>
#include <span>
#include <type_traits>
#include <vector>

#include "hephaistos/argument.hpp"
#include "hephaistos/bindings.hpp"
#include "hephaistos/buffer.hpp"
#include "hephaistos/command.hpp"
#include "hephaistos/imageformat.hpp"

namespace hephaistos {

namespace vulkan {
    struct Program;
}

/**
 * @brief List of subgroup properties and supported operations
*/
struct SubgroupProperties {
    /**
     * @brief Threads per subgroup
    */
    uint32_t subgroupSize;
    /**
     * @brief Support for GL_KHR_shader_subgroup_basic
    */
    bool basicSupport;
    /**
     * @brief Support for GL_KHR_shader_subgroup_vote
    */
    bool voteSupport;
    /**
     * @brief Support for GL_KHR_shader_subgroup_arithmetic
    */
    bool arithmeticSupport;
    /**
     * @brief Support for GL_KHR_shader_subgroup_ballot
    */
    bool ballotSupport;
    /**
     * @brief Support for GL_KHR_shader_subgroup_shuffle
    */
    bool shuffleSupport;
    /**
     * @brief Support for GL_KHR_shader_subgroup_shuffle_relative
    */
    bool shuffleRelativeSupport;
    /**
     * @brief Support for GL_KHR_shader_subgroup_clustered
    */
    bool shuffleClusteredSupport;
    /**
     * @brief Support for GL_KHR_shader_subgroup_quad
    */
    bool quadSupport;
    /**
     * @brief Support for GL_EXT_subgroup_uniform_control_flow
     */
    bool uniformControlFlowSupport;
    /**
     * @brief Support for GL_EXT_maximal_reconvergence
     */
    bool maximalReconvergenceSupport;
};
/**
 * @brief Returns the subgroup properties of the given device
 * 
 * @return Subgroup properties of the device
*/
[[nodiscard]] HEPHAISTOS_API SubgroupProperties
    getSubgroupProperties(const DeviceHandle& device);
/**
 * @brief Returns the subgroup properties of the device used to create the
 * given context
 * 
 * @return Subgroup properties of the context
*/
[[nodiscard]] HEPHAISTOS_API SubgroupProperties
    getSubgroupProperties(const ContextHandle& context);

/**
 * @brief Command for executing a program using the given group size 
*/
class HEPHAISTOS_API DispatchCommand : public Command {
public:
    /**
     * @brief Amount of groups in X dimension
    */
    uint32_t groupCountX;
    /**
     * @brief Amount of groups in Y dimension
    */
    uint32_t groupCountY;
    /**
     * @brief Amount of groups in Z dimension
    */
    uint32_t groupCountZ;
    /**
     * @brief Data used as push constant
     * 
     * Data used as push constant, e.g. a simple struct.
     * May be empty, i.e. zero sized, if no push data should be sent.
    */
    std::span<const std::byte> pushData;

    void record(vulkan::Command& cmd) const override;

    DispatchCommand(const DispatchCommand&);
    DispatchCommand& operator=(const DispatchCommand&);

    DispatchCommand(DispatchCommand&&) noexcept;
    DispatchCommand& operator=(DispatchCommand&&) noexcept;

    /**
     * @brief Creates a new DispatchCommand for the given program
     * 
     * @param program Program to dispatch
     * @param x Amount of groups in X dimension
     * @param y Amount of groups in Y dimension
     * @param z Amount of groups in Z dimension
     * @param push Data used as push constant
    */
    DispatchCommand(
        const vulkan::Program& program,
        const std::vector<VkWriteDescriptorSet> params,
        uint32_t x, uint32_t y, uint32_t z,
        std::span<const std::byte> push);
    ~DispatchCommand() override;

private:
    std::reference_wrapper<const vulkan::Program> program;
    std::vector<VkWriteDescriptorSet> params;
};

/**
 * @brief Data expected by the indirect dispatches
*/
struct DispatchIndirect {
    /**
     * @brief Amount of groups in X dimension
    */
    uint32_t groupCountX;
    /**
     * @brief Amount of groups in Y dimension
    */
    uint32_t groupCountY;
    /**
     * @brief Amount of groups in Z dimension
    */
    uint32_t groupCountZ;
};

/**
 * @brief Command for dispatching a program using params read from device memory
 *
 * Indirect dispatch allows to read the parameterization of a dispatch from a
 * tensor instead of statically define them during recording of the command.
 * This can be used to dynamically change it e.g. by a different program.
 * Expects to read data as defined by DispatchIndirect from the tensor at the
 * given offset.
 */
class HEPHAISTOS_API DispatchIndirectCommand : public Command {
public:
    /**
     * @brief Tensor from which to read the parametrization
    */
    std::reference_wrapper<const Tensor<std::byte>> tensor;
    /**
     * @brief Offset in bytes into tensor from which to start reading
    */
    uint64_t offset;
    /**
     * @brief Additional push data
     * 
     * Data used as push constant, e.g. a simple struct.
     * May be empty, i.e. zero sized, if no push data should be sent.
    */
    std::span<const std::byte> pushData;

    void record(vulkan::Command& cmd) const override;

    DispatchIndirectCommand(const DispatchIndirectCommand&);
    DispatchIndirectCommand& operator=(const DispatchIndirectCommand&);

    DispatchIndirectCommand(DispatchIndirectCommand&&) noexcept;
    DispatchIndirectCommand& operator=(DispatchIndirectCommand&&) noexcept;

    /**
     * @brief Creates a new DispatchIndirectCommand for the given program
     * 
     * @param program Program to dispatch
     * @param tensor Tensor from which to read the parametrization
     * @param offset Offset into the tensor from which to start reading
     * @param push Data used as push constant
    */
    DispatchIndirectCommand(
        const vulkan::Program& program,
        const std::vector<VkWriteDescriptorSet> params,
        const Tensor<std::byte>& tensor,
        uint64_t offset,
        std::span<const std::byte> push);
    ~DispatchIndirectCommand() override;

private:
    std::reference_wrapper<const vulkan::Program> program;
    std::vector<VkWriteDescriptorSet> params;
};

/**
 * @brief Shape of a workgroup
 * 
 * Dispatches are defined in units of workgroups which themselves contain
 * multiple threads arranged in three dimension.
*/
struct LocalSize {
    /**
     * @brief Threads in a single workgroup in X Dimension
    */
    uint32_t x;
    /**
     * @brief Threads in a single workgroup in Y Dimension
    */
    uint32_t y;
    /**
     * @brief Threads in a single workgroup in Z Dimension
    */
    uint32_t z;
};

/**
 * @brief Program containing shader code to run on the device.
 * 
 * Program encapsulates shader code to be run on the device. Manages a state of
 * bound parameters and allows to create commands for dispatching the program.
 * Provides basic methods for introspecting parameters defined by the shader.
*/
class HEPHAISTOS_API Program : public Resource, public BindingTarget {
public:
    /**
     * @brief Returns the thread shape of single workgroup
     * 
     * @return LocalSize of the program
    */
    [[nodiscard]] const LocalSize& getLocalSize() const noexcept;

    /**
     * @brief Dispatches the current program
     * 
     * @param push Additional push data
     * @param x Amount of groups in X dimension
     * @param y Amount of groups in Y dimension
     * @param z Amount of groups in Z dimension
     * @return DispatchCommand to issue a dispatch
    */
    [[nodiscard]] DispatchCommand dispatch(
        std::span<const std::byte> push,
        uint32_t x = 1, uint32_t y = 1, uint32_t z = 1) const;
    /**
     * @brief Dispatches the current program
     * 
     * @param x Amount of groups in X dimension
     * @param y Amount of groups in Y dimension
     * @param z Amount of groups in Z dimension
     * @return DispatchCommand to issue a dispatch
    */
    [[nodiscard]] DispatchCommand dispatch(
        uint32_t x = 1, uint32_t y = 1, uint32_t z = 1) const;
    /**
     * @brief Dispatches the current program
     * 
     * @param push Additional push data
     * @param x Amount of groups in X dimension
     * @param y Amount of groups in Y dimension
     * @param z Amount of groups in Z dimension
     * @return DispatchCommand to issue a dispatch
    */
    template<class T, typename = typename std::enable_if_t<std::is_standard_layout_v<T> && !std::is_integral_v<T>>>
    [[nodiscard]] DispatchCommand dispatch(const T& push, uint32_t x = 1, uint32_t y = 1, uint32_t z = 1) const {
        return dispatch({ reinterpret_cast<const std::byte*>(&push), sizeof(T)}, x, y, z);
    }

    /**
     * @brief Dispatches the current program using indirect params
     * 
     * @param push Additional push data
     * @param tensor Tensor from which to read the parameterization
     * @param offset Offset from which to start reading the parametrization
     * @return DispatchIndirectCommand to issue a indirect dispatch
    */
    [[nodiscard]] DispatchIndirectCommand dispatchIndirect(
        std::span<const std::byte> push, const Tensor<std::byte>& tensor, uint64_t offset = 0) const;
        /**
     * @brief Dispatches the current program using indirect params
     * 
     * @param tensor Tensor from which to read the parameterization
     * @param offset Offset from which to start reading the parametrization
     * @return DispatchIndirectCommand to issue a indirect dispatch
    */
    [[nodiscard]] DispatchIndirectCommand dispatchIndirect(
        const Tensor<std::byte>& tensor, uint64_t offset = 0) const;
    /**
     * @brief Dispatches the current program using indirect params
     * 
     * @param push Additional push data
     * @param tensor Tensor from which to read the parameterization
     * @param offset Offset from which to start reading the parametrization
     * @return DispatchIndirectCommand to issue a indirect dispatch
    */
    template<class T, typename = typename std::enable_if_t<std::is_standard_layout_v<T>>>
    [[nodiscard]] DispatchIndirectCommand dispatchIndirect(
        const T& push, const Tensor<std::byte>& tensor, uint64_t offset = 0) const
    {
        return dispatchIndirect({ reinterpret_cast<const std::byte*>(&push), sizeof(T) }, tensor, offset);
    }

    Program(const Program&) = delete;
    Program& operator=(const Program&) = delete;

    Program(Program&& other) noexcept;
    Program& operator=(Program&& other);

    /**
     * @brief Creates a new Program on the given context
     * 
     * @param context Context on which to create the program
     * @param code Compiled shader byte code
    */
    Program(ContextHandle context, std::span<const uint32_t> code);
    /**
     * @brief Creates a new Program on the given context
     * 
     * @param context Context on which to create the program
     * @param code Compiled shader byte code
     * @param specialization Data used to populate specialization constants
    */
    Program(ContextHandle context, std::span<const uint32_t> code, std::span<const std::byte> specialization);
    /**
     * @brief Creates a new Program on the given context
     * 
     * @param context Context on which to create the program
     * @param code Compiled shader byte code
     * @param specialization Data used to populate specialization constants
    */
    template<class T>
    Program(ContextHandle context, std::span<const uint32_t> code, const T& specialization)
        : Program(std::move(context), code, std::as_bytes(std::span<const T>{ &specialization, 1 }))
    {}
    ~Program() override;

protected: //internal

    void onDestroy() override;

private:
    std::unique_ptr<vulkan::Program> program;
};

/**
 * @brief Command for flushing device memory
 * 
 * This commands creates an explicit memory dependency between consecutive
 * program dispatches ensuring earlier writes are made visible to the following
 * dispatches.
*/
class HEPHAISTOS_API FlushMemoryCommand : public Command{
public:
    /**
     * @brief Context on which to insert the dependency
    */
    std::reference_wrapper<const vulkan::Context> context;

    void record(vulkan::Command& cmd) const override;

    FlushMemoryCommand(const FlushMemoryCommand&);
    FlushMemoryCommand& operator=(const FlushMemoryCommand&);

    FlushMemoryCommand(FlushMemoryCommand&& other) noexcept;
    FlushMemoryCommand& operator=(FlushMemoryCommand&& other) noexcept;

    /**
     * @brief Creates a new FlushMemoryCommand
     * 
     * @param context Context on which to create the dependency
    */
    explicit FlushMemoryCommand(const ContextHandle& context);
    ~FlushMemoryCommand() override;
};
/**
 * @brief Creates a new FlushMemoryCommand
 * 
 * @param context Context on which to create the dependency
 * @return FlushMemoryCommand creating the dependency on being recorded
*/
[[nodiscard]] inline FlushMemoryCommand flushMemory(const ContextHandle& context) {
    return FlushMemoryCommand(context);
}

}
