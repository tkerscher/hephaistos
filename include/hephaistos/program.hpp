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
 * @brief Parameter Type
 * 
 * Enumeration of parameter types a binding inside a program can define.
 * Used for reflections.
*/
enum class ParameterType {
    //SAMPLER = 0,
    
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

/**
 * @brief Image properties a binding defines
*/
struct ImageBindingTraits {
    /**
     * @brief Format of the image binding
    */
    ImageFormat format;
    /**
     * @brief Number of dimension the image has
    */
    uint8_t dims;
};

/**
 * @brief Properties of a binding inside a program
*/
struct BindingTraits {
    /**
     * @brief Name of the binding
     * 
     * Name of the binding as defined in the shader code.
     * Can be used to bind parameter to this binding in a program.
     * The compiler may have stripped away this information, where this becomes
     * an empty string.
    */
    std::string name;
    /**
     * @brief Binding number
     * 
     * Binding number, i.e. its location, as defined in the shader.
     * Can be used to bind parameter to this binding in a program and will
     * always be present.
    */
    uint32_t binding;
    /**
     * @brief Type of the binding
    */
    ParameterType type;
    /**
     * @brief Image properties of the binding
     * 
     * Only set if the binding type expects an image or texture
    */
    std::optional<ImageBindingTraits> imageTraits;
    /**
     * @brief Multiplicity of the binding
     * 
     * For plain bindings this will be one, whereas for array bindings, this
     * will give the size of the array. A value of zero denotes a runtime array.
    */
    uint32_t count;
};

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
class HEPHAISTOS_API Program : public Resource {
public:
    /**
     * @brief Returns the thread shape of single workgroup
     * 
     * @return LocalSize of the program
    */
    [[nodiscard]] const LocalSize& getLocalSize() const noexcept;

    /**
     * @brief Returns the traits of the given binding
     * 
     * Reflects the binding specified by its binding number in the shader code.
     * Throws if the given binding does not exist.
     * 
     * @param i Binding number to be reflected
     * @return BindingTraits of the given binding
    */
    [[nodiscard]] const BindingTraits& getBindingTraits(uint32_t i) const;
    /**
     * @brief Returns the traits of the given binding
     * 
     * Reflects the binding specified by its name as defined in the shader code.
     * Throws if the given binding cannot be found.
     * @note Shader compiler may strip away the binding names
     * 
     * @param name Name of the binding to be reflected
     * @return BindingTraits of the given binding
    */
    [[nodiscard]] const BindingTraits& getBindingTraits(std::string_view name) const;
    /**
     * @brief Checks whether the given binding is currently bound
     * 
     * @param i Binding number to be checked
     * @return True, if the binding is currently bound, false otherwise
     */
    [[nodiscard]] bool isBindingBound(uint32_t i) const;
    /**
     * @brief Checks wether the given binding is currently bound
     * 
     * Checks the binding specified by its name as defined in the shader code.
     * Throws if the given binding cannot be found.
     * @note Shader compiler may strip away the binding names
     * 
     * @param name Name of the binding to be checked
     * @return True, if the binding is currently bound, false otherwise
     */
    [[nodiscard]] bool isBindingBound(std::string_view name) const;
    /**
     * @brief Returns a list of BindingTraits of all bindings
     * 
     * @return Vector of BindingTraits one for each binding
    */
    [[nodiscard]] const std::vector<BindingTraits>& listBindings() const noexcept;

    /**
     * @brief Binds the given parameter
     * 
     * @param param Parameter to bind
     * @param binding Binding number to bind the parameter to
    */
    template<class T>
    void bindParameter(const T& param, uint32_t binding) {
        param.bindParameter(getBinding(binding));
    }
    /**
     * @brief Binds the given parameter
     * 
     * @param param Parameter to bind
     * @param binding Name of the binding to bind the parameter to
    */
    template<class T>
    void bindParameter(const T& param, std::string_view binding) {
        param.bindParameter(getBinding(binding));
    }
    /**
     * @brief Binds the list of parameters
     * 
     * Binds the given list of parameters to bindings in the order as they are
     * defined in the program.
     * 
     * @param param... List of parameters to bind
    */
    template<class ...T>
    void bindParameterList(const T&...param) {
        uint32_t binding = 0;
        (param.bindParameter(getBinding(binding++)),...);
    }

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
    Program& operator=(Program&& other) noexcept;

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

public: //internal
    [[nodiscard]] VkWriteDescriptorSet& getBinding(uint32_t i);
    [[nodiscard]] VkWriteDescriptorSet& getBinding(std::string_view name);

private:
    std::unique_ptr<vulkan::Program> program;
    std::vector<BindingTraits> bindingTraits;
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
