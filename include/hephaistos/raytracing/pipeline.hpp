#pragma once

#include <span>
#include <variant>

#include "hephaistos/bindings.hpp"
#include "hephaistos/buffer.hpp"
#include "hephaistos/command.hpp"
#include "hephaistos/handles.hpp"

namespace hephaistos {

namespace vulkan {
    struct RayTracingPipeline;
}

/**
 * @brief References a region inside a shader binding table
 */
struct ShaderBindingTableRegion {
    /**
     * @brief Address at the start of the region
     */
    uint64_t address = 0;
    /**
     * @brief Stride between entries in the table
     */
    uint32_t stride = 0;
    /**
     * @brief Amount of referenced entries
     */
    uint32_t count = 0;
};

/**
 * @brief Single entry in a shader binding table
 */
struct ShaderBindingTableEntry {
    /**
     * @brief Index of the referenced shader
     *
     * Index of the referenced shader as passed to the ray tracing pipeline
     * or ~0u for an empty entry.
     */
    uint32_t groupIndex;
    /**
     * @brief Optional additional data accessible by the shader as shader record.
     */
    std::span<const std::byte> shaderRecord = {};
};

/**
 * @brief Shader binding table
 * 
 * Shader binding table acts as a sort of vtable for ray tracing pipelines.
 * See the Vulkan specification for a more detailed explanation.
 */
class HEPHAISTOS_API ShaderBindingTable : public Resource {
public:
    operator ShaderBindingTableRegion() const;

    ShaderBindingTable(const ShaderBindingTable&) = delete;
    ShaderBindingTable& operator=(const ShaderBindingTable&) = delete;

    ShaderBindingTable(ShaderBindingTable&&) noexcept;
    ShaderBindingTable& operator=(ShaderBindingTable&&) noexcept;

    virtual ~ShaderBindingTable();

public: //internal
    void onDestroy() override;

private:
    ShaderBindingTable(
        ContextHandle context, BufferHandle buffer,
        uint32_t stride, uint32_t count);
    friend class RayTracingPipeline;

private:
    BufferHandle buffer;
    ShaderBindingTableRegion region;
};

/**
 * @brief Compiled shader code reference
*/
struct ShaderCode {
    /**
     * @brief Compiled SPIR-V code
     */
    std::span<const uint32_t> code = {};
    /**
     * @brief Name of code entry point
     */
    std::string_view entryName = "main";
};

/**
 * @brief Ray generation shader used by ray tracing pipelines
 */
struct RayGenerateShader {
    ShaderCode code;
};
/**
 * @brief Ray miss shader used by ray tracing pipelines
 */
struct RayMissShader {
    ShaderCode code;
};
/**
 * @brief Ray hit shader used by ray tracing pipelines
 */
struct RayHitShader {
    ShaderCode closest;
    ShaderCode any;
};
/**
 * @brief Callable shader used by ray tracing pipelines
 */
struct CallableShader {
    ShaderCode code;
};
/**
 * @brief Variant combining all supported ray tracing shader types
 */
using RayTracingShader = std::variant<
    RayGenerateShader,
    RayMissShader,
    RayHitShader,
    CallableShader
>;

/**
 * @brief Amount of rays to dispatch in each dimension
 */
struct RayCount {
    uint32_t x = 1;
    uint32_t y = 1;
    uint32_t z = 1;
};

/**
 * @brief Composition of all required shaders when tracing rays
 */
struct ShaderBindings {
    /**
     * @brief Ray generation shader to use
     */
    ShaderBindingTableRegion rayGenShaders;
    /**
     * @brief Miss shaders to use
     */
    ShaderBindingTableRegion missShaders;
    /**
     * @brief Hit shaders to use
     */
    ShaderBindingTableRegion hitShaders;
    /**
     * @brief Optional callable shaders to use
     */
    ShaderBindingTableRegion callableShaders = {};
};

/**
 * @brief Command for issuing rays to be traced using a ray tracing pipeline
 */
class HEPHAISTOS_API TraceRaysCommand : public Command {
public:
    /**
     * @brief Shaders to be used to trace rays
     */
    ShaderBindings shaderBindings;
    /**
     * @brief Amount of rays to trace
     */
    RayCount rayCount;
    /**
     * @brief Data used as push constant
     * 
     * Data used as push constant, e.g. a simple struct.
     * May be empty, i.e. zero sized, if no push data should be sent.
    */
    std::span<const std::byte> pushData;

    void record(vulkan::Command& cmd) const override;

    TraceRaysCommand(const TraceRaysCommand&);
    TraceRaysCommand& operator=(const TraceRaysCommand&);

    TraceRaysCommand(TraceRaysCommand&&) noexcept;
    TraceRaysCommand& operator=(TraceRaysCommand&&) noexcept;

    /**
     * @brief Creates a new TraceRaysCommand
     * 
     * @param pipeline Ray tracing pipeline to be used
     * @param params List of parameters to use when tracing rays
     * @param shaderBindings Shader to be used for tracing rays
     * @param rayCount Amount of rays to trace
     * @param push Optional data used as push constant
     */
    TraceRaysCommand(
        const vulkan::RayTracingPipeline& pipeline,
        const std::vector<VkWriteDescriptorSet> params,
        ShaderBindings shaderBindings,
        RayCount rayCount,
        std::span<const std::byte> push);
    ~TraceRaysCommand() override;

private:
    std::reference_wrapper<const vulkan::RayTracingPipeline> pipeline;
    std::vector<VkWriteDescriptorSet> params;
};

/**
 * @brief Command for tracing rays where the amount is read from device memory
 * 
 * Indirect trace allows to read the amount of rays to trace from device memory
 * allowing to dynamically set the amount in a previous shader. The shader
 * themselves, that is ray generation, miss, hit and callable shaders, cannot be
 * changed.
 */
class HEPHAISTOS_API TraceRaysIndirectCommand : public Command {
public:
    /**
     * @brief Shaders to be used to trace rays
     */
    ShaderBindings shaderBindings;
    /**
     * @brief Tensor from which to read the amount of rays to trace
    */
    std::reference_wrapper<const Tensor<std::byte>> tensor;
    /**
     * @brief Offset in bytes into tensor from which to start reading
    */
    uint64_t offset;
    /**
     * @brief Data used as push constant
     * 
     * Data used as push constant, e.g. a simple struct.
     * May be empty, i.e. zero sized, if no push data should be sent.
    */
    std::span<const std::byte> pushData;

    void record(vulkan::Command& command) const override;

    TraceRaysIndirectCommand(const TraceRaysIndirectCommand&);
    TraceRaysIndirectCommand& operator=(const TraceRaysIndirectCommand&);

    TraceRaysIndirectCommand(TraceRaysIndirectCommand&&) noexcept;
    TraceRaysIndirectCommand& operator=(TraceRaysIndirectCommand&&) noexcept;

    /**
     * @brief Creates a new TraceRaysIndirectCommand
     * 
     * @param pipeline Ray tracing pipeline to be used for tracing rays
     * @param params List of parameters to use when tracing
     * @param bindings Shader to use when tracing rays
     * @param tensor Tensor from which to read the amount of rays to trace
     * @param offset Offset in bytes into the tensor at which to start reading
     * @param push Optional data used as push constant
     */
    TraceRaysIndirectCommand(
        const vulkan::RayTracingPipeline& pipeline,
        std::vector<VkWriteDescriptorSet> params,
        ShaderBindings bindings,
        const Tensor<std::byte>& tensor,
        uint64_t offset,
        std::span<const std::byte> push);
    ~TraceRaysIndirectCommand() override;

private:
    std::reference_wrapper<const vulkan::RayTracingPipeline> pipeline;
    std::vector<VkWriteDescriptorSet> params;
};

/**
 * @brief Pipeline responsible for tracing rays
 * 
 * Ray tracing pipelines hold a collection of smaller shader stages needed for
 * tracing rays, namely the ray generation, miss, hit and callable shaders.
 * These can be arranged as needed, e.g. changing the ray generation shader or
 * using multiple different hit shaders. These arrangements are stored in shader
 * binding tables, which can hold additional data for these shaders allowing to
 * parameterize the referenced shaders.
 */
class HEPHAISTOS_API RayTracingPipeline : public Resource, public BindingTarget {
public:
    /**
     * @brief Returns total amount of shaders in this pipeline.
     */
    [[nodiscard]] uint32_t shaderCount() const noexcept;

    /**
     * @brief Creates a new shader binding table
     * 
     * Creates a new shader binding table from the given range of shader groups.
     * 
     * @param firstGroupIdx Start of the shader group range
     * @param count Count of shader groups in the range
     */
    [[nodiscard]] ShaderBindingTable createShaderBindingTable(
        uint32_t firstGroupIdx, uint32_t count = 1) const;
    /**
     * @brief Creates a new shader binding table
     * 
     * Creates a new shader binding table with the specified entries. Each entry
     * references a shader group by its index and optionally additional data.
     * 
     * @param entries Entries making up the shader binding table
     */
    [[nodiscard]] ShaderBindingTable createShaderBindingTable(
        std::span<const ShaderBindingTableEntry> entries) const;

    /**
     * @brief Traces the given amounts of rays
     * 
     * @param bindings Shader to be used for tracing
     * @param groupCount Amount of rays to trace
     * @param push Optional data used as push constant
     * @return TraceRaysCommand to issue tracing rays
     */
    [[nodiscard]] TraceRaysCommand traceRays(
        ShaderBindings bindings,
        RayCount rayCount,
        std::span<const std::byte> push = {}) const;
    /**
     * @brief Traces the given amounts of rays
     * 
     * @param bindings Shader to be used for tracing
     * @param groupCount Amount of rays to trace
     * @param push Optional data used as push constant
     * @return TraceRaysCommand to issue tracing rays
     */
    template<class T,
        typename = typename std::enable_if_t<
            std::is_standard_layout_v<T> && !std::is_integral_v<T>
        >>
    [[nodiscard]] TraceRaysCommand traceRays(
        ShaderBindings bindings,
        RayCount rayCount,
        const T& push) const
    {
        return traceRays(
            bindings, rayCount,
            { reinterpret_cast<const std::byte*>(&push), sizeof(T) });
    }

    /**
     * @brief Traces rays using the amount stored in the given tensor
     * 
     * @param bindings Shaders to be used for tracing
     * @param tensor Tensor from which to read the amount of rays
     * @param offset Offset in bytes into the tensor at which to start reading
     * @return TraceRaysIndirectCommand to issue the tracing
     */
    [[nodiscard]] TraceRaysIndirectCommand traceRaysIndirect(
        ShaderBindings bindings,
        const Tensor<std::byte>& tensor,
        uint64_t offset = 0) const;
    /**
     * @brief Traces rays using the amount stored in the given tensor
     * 
     * @param bindings Shaders to be used for tracing
     * @param push Optional data used for push constant
     * @param tensor Tensor from which to read the amount of rays
     * @param offset Offset in bytes into the tensor at which to start reading
     * @return TraceRaysIndirectCommand to issue the tracing
     */
    [[nodiscard]] TraceRaysIndirectCommand traceRaysIndirect(
        ShaderBindings bindings,
        std::span<const std::byte> push,
        const Tensor<std::byte>& tensor,
        uint64_t offset = 0) const;
    /**
     * @brief Traces rays using the amount stored in the given tensor
     * 
     * @param bindings Shaders to be used for tracing
     * @param push Optional data used for push constant
     * @param tensor Tensor from which to read the amount of rays
     * @param offset Offset in bytes into the tensor at which to start reading
     * @return TraceRaysIndirectCommand to issue the tracing
     */
    template<class T, typename = typename std::enable_if_t<std::is_standard_layout_v<T>>>
    [[nodiscard]] TraceRaysIndirectCommand traceRaysIndirect(
        ShaderBindings bindings,
        const T& push,
        const Tensor<std::byte>& tensor,
        uint64_t offset = 0) const
    {
        return traceRaysIndirect(
            { reinterpret_cast<const std::byte*>(&push), sizeof(T) },
            tensor, offset);
    }
    
    RayTracingPipeline(const RayTracingPipeline&) = delete;
    RayTracingPipeline& operator=(const RayTracingPipeline&) = delete;

    RayTracingPipeline(RayTracingPipeline&&) noexcept;
    RayTracingPipeline& operator=(RayTracingPipeline&&) noexcept;

    /**
     * @brief Creates a new ray tracing pipeline
     * 
     * @param context Context on which to create the ray tracing pipeline
     * @param shader List of shaders to include
     * @param specialization Data used to populate specialization constants
     * @param maxRecursionDepth Max ray recursion depth
     */
    RayTracingPipeline(
        ContextHandle context,
        std::span<const RayTracingShader> shader,
        std::span<const std::byte> specialization,
        uint32_t maxRecursionDepth = 1
    );
    /**
     * @brief Creates a new ray tracing pipeline
     * 
     * @param context Context on which to create the ray tracing pipeline
     * @param shader List of shaders to include
     * @param maxRecursionDepth Max ray recursion depth
     */
    RayTracingPipeline(
        ContextHandle context,
        std::span<const RayTracingShader> shader,
        uint32_t maxRecursionDepth = 1
    );
    ~RayTracingPipeline() override;

protected: //internal
    void onDestroy() override;

private:
    std::unique_ptr<vulkan::RayTracingPipeline> pipeline;

    std::vector<std::byte> handleStorage;
    uint32_t handleSize;
    uint32_t handleCount;
};

}
