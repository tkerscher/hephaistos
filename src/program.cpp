#include "hephaistos/program.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include "volk.h"
#include "spirv_reflect.h"

#include "vk/reflection.hpp"
#include "vk/result.hpp"
#include "vk/types.hpp"

namespace hephaistos {

/********************************** SUBGROUP **********************************/

namespace {

SubgroupProperties getSubgroupProperties(VkPhysicalDevice device) {
    //fetch subgroup properties
    VkPhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR controlFlow{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_UNIFORM_CONTROL_FLOW_FEATURES_KHR
    };
    VkPhysicalDeviceShaderMaximalReconvergenceFeaturesKHR reconvergence{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MAXIMAL_RECONVERGENCE_FEATURES_KHR,
        .pNext = &controlFlow
    };
    VkPhysicalDeviceFeatures2 features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &reconvergence
    };
    vkGetPhysicalDeviceFeatures2(device, &features);
    VkPhysicalDeviceSubgroupProperties subgroupProps{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
    };
    VkPhysicalDeviceProperties2 props{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &subgroupProps
    };
    vkGetPhysicalDeviceProperties2(device, &props);

    //build struct
    return SubgroupProperties{
        .subgroupSize            = subgroupProps.subgroupSize,
        .basicSupport            = !!(subgroupProps.supportedOperations & VK_SUBGROUP_FEATURE_BASIC_BIT),
        .voteSupport             = !!(subgroupProps.supportedOperations & VK_SUBGROUP_FEATURE_VOTE_BIT),
        .arithmeticSupport       = !!(subgroupProps.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT),
        .ballotSupport           = !!(subgroupProps.supportedOperations & VK_SUBGROUP_FEATURE_BALLOT_BIT),
        .shuffleSupport          = !!(subgroupProps.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT),
        .shuffleRelativeSupport  = !!(subgroupProps.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT),
        .shuffleClusteredSupport = !!(subgroupProps.supportedOperations & VK_SUBGROUP_FEATURE_CLUSTERED_BIT),
        .quadSupport             = !!(subgroupProps.supportedOperations & VK_SUBGROUP_FEATURE_QUAD_BIT),
        .uniformControlFlowSupport   = controlFlow.shaderSubgroupUniformControlFlow == VK_TRUE,
        .maximalReconvergenceSupport = reconvergence.shaderMaximalReconvergence == VK_TRUE
    };
}

}

SubgroupProperties getSubgroupProperties(const DeviceHandle& device) {
    return getSubgroupProperties(device->device);
}
SubgroupProperties getSubgroupProperties(const ContextHandle& context) {
    return getSubgroupProperties(context->physicalDevice);
}

/********************************** DISPATCH **********************************/

namespace vulkan {

struct Program {
    VkDescriptorSetLayout descriptorSetLayout = nullptr;
    VkPipelineLayout pipeLayout = nullptr;
    VkPipeline pipeline = nullptr;
    uint32_t set = 0;

    LocalSize localSize{};

    const Context& context;

    Program(const Context& context)
        : context(context)
    {}
};

}

void DispatchCommand::record(vulkan::Command& cmd) const {
    auto& prog = program.get();
    auto& context = prog.context;

    //bind pipeline
    context.fnTable.vkCmdBindPipeline(cmd.buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        prog.pipeline);

    //bind params if there are any
    if (!params.empty()) {
        context.fnTable.vkCmdPushDescriptorSetKHR(cmd.buffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            prog.pipeLayout,
            prog.set,
            static_cast<uint32_t>(params.size()),
            params.data());
    }
    //push constant if there is any
    if (!pushData.empty()) {
        context.fnTable.vkCmdPushConstants(cmd.buffer,
            prog.pipeLayout,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            static_cast<uint32_t>(pushData.size_bytes()),
            pushData.data());
    }

    //dispatch
    context.fnTable.vkCmdDispatch(cmd.buffer,
        groupCountX, groupCountY, groupCountZ);
}

DispatchCommand::DispatchCommand(const DispatchCommand&) = default;
DispatchCommand& DispatchCommand::operator=(const DispatchCommand&) = default;

DispatchCommand::DispatchCommand(DispatchCommand&&) noexcept = default;
DispatchCommand& DispatchCommand::operator=(DispatchCommand&&) noexcept = default;

DispatchCommand::DispatchCommand(
    const vulkan::Program& program,
    const std::vector<VkWriteDescriptorSet> params,
    uint32_t x, uint32_t y, uint32_t z,
    std::span<const std::byte> push
)
    : groupCountX(x)
    , groupCountY(y)
    , groupCountZ(z)
    , pushData(push)
    , program(std::cref(program))
    , params(params)
{}
DispatchCommand::~DispatchCommand() = default;

void DispatchIndirectCommand::record(vulkan::Command& cmd) const {
    auto buffer = tensor.get().getBuffer().buffer;
    auto& prog = program.get();
    auto& context = prog.context;

    //bind pipeline
    context.fnTable.vkCmdBindPipeline(cmd.buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        prog.pipeline);

    //bind params if there are any
    if (!params.empty()) {
        context.fnTable.vkCmdPushDescriptorSetKHR(cmd.buffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            prog.pipeLayout,
            prog.set,
            static_cast<uint32_t>(params.size()),
            params.data());
    }
    //push constant if there is any
    if (!pushData.empty()) {
        context.fnTable.vkCmdPushConstants(cmd.buffer,
            prog.pipeLayout,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            static_cast<uint32_t>(pushData.size_bytes()),
            pushData.data());
    }

    //barrier to ensure indirect data is complete
    VkBufferMemoryBarrier2 barrier{
        .sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask        = VK_PIPELINE_STAGE_2_TRANSFER_BIT |
                               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask       = VK_ACCESS_2_TRANSFER_WRITE_BIT |
                               VK_ACCESS_2_SHADER_WRITE_BIT,
        .dstStageMask        = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
        .dstAccessMask       = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer              = buffer,
        .offset              = offset,
        .size                = 12 // 3 * int
    };
    VkDependencyInfo depInfo{
        .sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers    = &barrier
    };
    context.fnTable.vkCmdPipelineBarrier2(cmd.buffer, &depInfo);

    //disptach indirect
    context.fnTable.vkCmdDispatchIndirect(cmd.buffer, buffer, offset);
}

DispatchIndirectCommand::DispatchIndirectCommand(const DispatchIndirectCommand&) = default;
DispatchIndirectCommand& DispatchIndirectCommand::operator=(const DispatchIndirectCommand&) = default;

DispatchIndirectCommand::DispatchIndirectCommand(DispatchIndirectCommand&&) noexcept = default;
DispatchIndirectCommand& DispatchIndirectCommand::operator=(DispatchIndirectCommand&&) noexcept = default;

DispatchIndirectCommand::DispatchIndirectCommand(
    const vulkan::Program& program,
    const std::vector<VkWriteDescriptorSet> params,
    const Tensor<std::byte>& tensor,
    uint64_t offset,
    std::span<const std::byte> push)
    : tensor(std::cref(tensor))
    , offset(offset)
    , pushData(push)
    , program(std::cref(program))
    , params(params)
{}
DispatchIndirectCommand::~DispatchIndirectCommand() = default;

/*********************************** PROGRAM **********************************/

const LocalSize& Program::getLocalSize() const noexcept {
    return program->localSize;
}

DispatchCommand Program::dispatch(std::span<const std::byte> push, uint32_t x, uint32_t y, uint32_t z) const {
    checkAllBindingsBound();
    return DispatchCommand(*program, boundParams, x, y, z, push);
}
DispatchCommand Program::dispatch(uint32_t x, uint32_t y, uint32_t z) const {
    return dispatch({}, x, y, z);
}

DispatchIndirectCommand Program::dispatchIndirect(
    std::span<const std::byte> push, const Tensor<std::byte>& tensor, uint64_t offset) const
{
    checkAllBindingsBound();
    return DispatchIndirectCommand(*program, boundParams, tensor, offset, push);
}
DispatchIndirectCommand Program::dispatchIndirect(const Tensor<std::byte>& tensor, uint64_t offset) const {
    checkAllBindingsBound();
    return DispatchIndirectCommand(*program, boundParams, tensor, offset, {});
}

void Program::onDestroy() {
    if (program) {
        auto& context = getContext();
        context->fnTable.vkDestroyPipeline(context->device, program->pipeline, nullptr);
        context->fnTable.vkDestroyPipelineLayout(context->device, program->pipeLayout, nullptr);
        context->fnTable.vkDestroyDescriptorSetLayout(context->device, program->descriptorSetLayout, nullptr);

        program.reset();
        bindingTraits.clear();
        boundParams.clear();
    }
}

Program::Program(Program&&) noexcept = default;
Program& Program::operator=(Program&&) = default;

Program::Program(ContextHandle context, std::span<const uint32_t> code, std::span<const std::byte> specialization)
    : Resource(std::move(context))
    , BindingTarget()
    , program(std::make_unique<vulkan::Program>(*getContext()))
{
    //context reference
    auto& con = *getContext();

    //reflect shader
    vulkan::LayoutReflectionBuilder reflection;
    reflection.add(code);

    //create pipeline layout
    reflection.createDescriptorSetLayout(con, &program->descriptorSetLayout);
    reflection.createPipelineLayout(con, &program->descriptorSetLayout, &program->pipeLayout);
    //fetch specialization map
    auto specMap = reflection.createSpecializationMap(specialization.size_bytes() / 4);
    auto specInfo = vulkan::createSpecializationInfo(specMap, specialization);
    //move params
    bindingTraits = std::move(reflection.traits);
    boundParams = std::move(reflection.params);
    program->localSize = {
        reflection.localSize.x,
        reflection.localSize.y,
        reflection.localSize.z
    };

    //create compute pipeline
    VkShaderModuleCreateInfo shaderInfo{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = code.size_bytes(),
        .pCode = code.data()
    };
    VkPipelineShaderStageCreateInfo stageInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = &shaderInfo,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .pName = "main",
        .pSpecializationInfo = specMap.empty() ? nullptr : &specInfo
    };
    VkComputePipelineCreateInfo pipeInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = stageInfo,
        .layout = program->pipeLayout
    };
    vulkan::checkResult(con.fnTable.vkCreateComputePipelines(
        con.device, con.cache,
        1, &pipeInfo, nullptr,
        &program->pipeline));
}
Program::Program(ContextHandle context, std::span<const uint32_t> code)
    : Program(std::move(context), code, {})
{}
Program::~Program() {
    onDestroy();
}

/********************************* FLUSH MEMORY *******************************/

void FlushMemoryCommand::record(vulkan::Command& cmd) const {
    auto compStages = context.get().computeStages;

    VkMemoryBarrier2 barrier{
        .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask  = compStages,
        .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
        .dstStageMask  = compStages,
        .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT,
    };
    VkDependencyInfo depInfo{
        .sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = 1,
        .pMemoryBarriers    = &barrier
    };
    context.get().fnTable.vkCmdPipelineBarrier2(cmd.buffer, &depInfo);
}

FlushMemoryCommand::FlushMemoryCommand(const FlushMemoryCommand&) = default;
FlushMemoryCommand& FlushMemoryCommand::operator=(const FlushMemoryCommand&) = default;

FlushMemoryCommand::FlushMemoryCommand(FlushMemoryCommand&&) noexcept = default;
FlushMemoryCommand& FlushMemoryCommand::operator=(FlushMemoryCommand&&) noexcept = default;

FlushMemoryCommand::FlushMemoryCommand(const ContextHandle& context)
    : context(*context)
{}
FlushMemoryCommand::~FlushMemoryCommand() = default;

}
