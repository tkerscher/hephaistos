#include "hephaistos/program.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include "volk.h"
#include "spirv_reflect.h"

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
    VkShaderModule shader = nullptr;
    VkPipeline pipeline = nullptr;
    uint32_t set = 0;

    LocalSize localSize{};

    std::vector<VkWriteDescriptorSet> boundParams;

    const Context& context;

    Program(const Context& context)
        : context(context)
    {}
};

bool isDescriptorSetEmpty(const VkWriteDescriptorSet& set) {
    return set.pNext == nullptr &&
        set.pImageInfo == nullptr &&
        set.pBufferInfo == nullptr &&
        set.pTexelBufferView == nullptr;
}

void checkAllBound(const std::vector<VkWriteDescriptorSet> boundParams) {
    if (std::any_of(
        boundParams.begin(),
        boundParams.end(),
        vulkan::isDescriptorSetEmpty)
    ) {
        std::ostringstream stream;
        stream << "Cannot dispatch program due to unbound bindings: ";
        //collect all unbound bindings to make the error more usefull
        auto i = 0u;
        for (auto& param : boundParams) {
            if (vulkan::isDescriptorSetEmpty(param))
                stream << i << " ";
            ++i;
        }

        throw std::logic_error(stream.str());
    }
}

}

void DispatchCommand::record(vulkan::Command& cmd) const {
    auto& prog = program.get();
    auto& context = prog.context;

    //we're working in the compute stage
    cmd.stage |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

    //bind pipeline
    context.fnTable.vkCmdBindPipeline(cmd.buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        prog.pipeline);

    //bind params
    context.fnTable.vkCmdPushDescriptorSetKHR(cmd.buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        prog.pipeLayout,
        prog.set,
        static_cast<uint32_t>(params.size()),
        params.data());

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
    uint32_t x, uint32_t y, uint32_t z,
    std::span<const std::byte> push
)
    : groupCountX(x)
    , groupCountY(y)
    , groupCountZ(z)
    , pushData(push)
    , program(std::cref(program))
    , params(program.boundParams)
{
    //sanity check: all params are bound (will throw if not)
    vulkan::checkAllBound(program.boundParams);
}
DispatchCommand::~DispatchCommand() = default;

void DispatchIndirectCommand::record(vulkan::Command& cmd) const {
    auto buffer = tensor.get().getBuffer().buffer;
    auto& prog = program.get();
    auto& context = prog.context;

    //we're working in the compute stage
    cmd.stage |=
        VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT |
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

    //bind pipeline
    context.fnTable.vkCmdBindPipeline(cmd.buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        prog.pipeline);

    //bind params
    context.fnTable.vkCmdPushDescriptorSetKHR(cmd.buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        prog.pipeLayout,
        prog.set,
        static_cast<uint32_t>(params.size()),
        params.data());

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
    VkBufferMemoryBarrier barrier{
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
        .buffer = buffer,
        .offset = offset,
        .size = 12 // 3 * int
    };
    context.fnTable.vkCmdPipelineBarrier(cmd.buffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0,
        0, nullptr,
        1, &barrier,
        0, nullptr);

    //disptach indirect
    context.fnTable.vkCmdDispatchIndirect(cmd.buffer, buffer, offset);
}

DispatchIndirectCommand::DispatchIndirectCommand(const DispatchIndirectCommand&) = default;
DispatchIndirectCommand& DispatchIndirectCommand::operator=(const DispatchIndirectCommand&) = default;

DispatchIndirectCommand::DispatchIndirectCommand(DispatchIndirectCommand&&) noexcept = default;
DispatchIndirectCommand& DispatchIndirectCommand::operator=(DispatchIndirectCommand&&) noexcept = default;

DispatchIndirectCommand::DispatchIndirectCommand(
    const vulkan::Program& program,
    const Tensor<std::byte>& tensor,
    uint64_t offset,
    std::span<const std::byte> push)
    : tensor(std::cref(tensor))
    , offset(offset)
    , pushData(push)
    , program(std::cref(program))
    , params(program.boundParams)
{
    //sanity check: all params are bound (will throw if not)
    vulkan::checkAllBound(program.boundParams);
}
DispatchIndirectCommand::~DispatchIndirectCommand() = default;

/*********************************** PROGRAM **********************************/

namespace {

#define STR(c) case SPV_REFLECT_RESULT_##c: return "Reflect: "#c;

inline auto printSpvResult(SpvReflectResult result) {
    switch (result) {
        STR(SUCCESS)
        STR(NOT_READY)
        STR(ERROR_PARSE_FAILED)
        STR(ERROR_ALLOC_FAILED)
        STR(ERROR_RANGE_EXCEEDED)
        STR(ERROR_NULL_POINTER)
        STR(ERROR_INTERNAL_ERROR)
        STR(ERROR_COUNT_MISMATCH)
        STR(ERROR_ELEMENT_NOT_FOUND)
        STR(ERROR_SPIRV_INVALID_CODE_SIZE)
        STR(ERROR_SPIRV_INVALID_MAGIC_NUMBER)
        STR(ERROR_SPIRV_UNEXPECTED_EOF)
        STR(ERROR_SPIRV_INVALID_ID_REFERENCE)
        STR(ERROR_SPIRV_SET_NUMBER_OVERFLOW)
        STR(ERROR_SPIRV_INVALID_STORAGE_CLASS)
        STR(ERROR_SPIRV_RECURSION)
        STR(ERROR_SPIRV_INVALID_INSTRUCTION)
        STR(ERROR_SPIRV_UNEXPECTED_BLOCK_DATA)
        STR(ERROR_SPIRV_INVALID_BLOCK_MEMBER_REFERENCE)
        STR(ERROR_SPIRV_INVALID_ENTRY_POINT)
        STR(ERROR_SPIRV_INVALID_EXECUTION_MODE)
    default:
        return "Unknown result code";
    }
}

#undef STR

void spvCheckResult(SpvReflectResult result) {
    if (result != SPV_REFLECT_RESULT_SUCCESS)
        throw std::runtime_error(printSpvResult(result));
}

ImageFormat castImageFormat(SpvImageFormat format) {
    switch (format) {
    case SpvImageFormatRgba32f:
        return ImageFormat::R32G32B32A32_SFLOAT;
    case SpvImageFormatRgba32i:
        return ImageFormat::R32G32B32A32_SINT;
    case SpvImageFormatRgba32ui:
        return ImageFormat::R32G32B32A32_UINT;
    case SpvImageFormatRg32f:
        return ImageFormat::R32G32_SFLOAT;
    case SpvImageFormatRg32i:
        return ImageFormat::R32G32_SINT;
    case SpvImageFormatRg32ui:
        return ImageFormat::R32G32_UINT;
    case SpvImageFormatR32f:
        return ImageFormat::R32G32_SFLOAT;
    case SpvImageFormatR32i:
        return ImageFormat::R32G32_SINT;
    case SpvImageFormatR32ui:
        return ImageFormat::R32G32B32A32_UINT;
    case SpvImageFormatRgba16i:
        return ImageFormat::R16G16B16A16_SINT;
    case SpvImageFormatRgba16ui:
        return ImageFormat::R16G16B16A16_UINT;
    case SpvImageFormatRgba8:
        return ImageFormat::R8G8B8A8_UNORM;
    case SpvImageFormatRgba8Snorm:
        return ImageFormat::R8G8B8A8_SNORM;
    case SpvImageFormatRgba8i:
        return ImageFormat::R8G8B8A8_SINT;
    case SpvImageFormatRgba8ui:
        return ImageFormat::R8G8B8A8_UINT;
    default:
        return ImageFormat::UNKNOWN;
    }
}

uint8_t castDimension(SpvDim dim) {
    switch (dim) {
    case SpvDim1D:
        return 1;
    case SpvDim2D:
        return 2;
    case SpvDim3D:
        return 3;
    default:
        return 0; //Unknown
    }
}

}

const LocalSize& Program::getLocalSize() const noexcept {
    return program->localSize;
}

const BindingTraits& Program::getBindingTraits(uint32_t i) const {
    if (i >= bindingTraits.size())
        throw std::runtime_error("There is no binding point at specified number!");

    return bindingTraits[i];
}
const BindingTraits& Program::getBindingTraits(std::string_view name) const {
    const auto& b = bindingTraits;
    auto it = std::find_if(b.begin(), b.end(), [name](const BindingTraits& t) -> bool {
        return t.name == name;
    });

    if (it == b.end())
        throw std::runtime_error("There is no binding point at specified location!");

    return *it;
}

bool Program::isBindingBound(uint32_t i) const {
    if (i >= bindingTraits.size())
        throw std::runtime_error("There is no binding point at specified number!");

    return !vulkan::isDescriptorSetEmpty(program->boundParams[i]);
}

bool Program::isBindingBound(std::string_view name) const {
    const auto& b = bindingTraits;
    auto it = std::find_if(b.begin(), b.end(), [name](const BindingTraits& t) -> bool {
        return t.name == name;
        });

    if (it == b.end())
        throw std::runtime_error("There is no binding point at specified location! Binding name: " + std::string(name));

    auto i = std::distance(b.begin(), it);
    return !vulkan::isDescriptorSetEmpty(program->boundParams[i]);
}

const std::vector<BindingTraits>& Program::listBindings() const noexcept {
    return bindingTraits;
}

VkWriteDescriptorSet& Program::getBinding(uint32_t i) {
    if (i >= program->boundParams.size())
        throw std::runtime_error("There is no binding point at specified number! Binding: " + std::to_string(i));

    return program->boundParams[i];
}
VkWriteDescriptorSet& Program::getBinding(std::string_view name) {
    const auto& b = bindingTraits;
    auto it = std::find_if(b.begin(), b.end(), [name](const BindingTraits& t) -> bool {
        return t.name == name;
    });

    if (it == b.end())
        throw std::runtime_error("There is no binding point at specified location! Binding name: " + std::string(name));

    auto i = std::distance(b.begin(), it);
    return program->boundParams[i];
}

DispatchCommand Program::dispatch(std::span<const std::byte> push, uint32_t x, uint32_t y, uint32_t z) const {
    return DispatchCommand(*program, x, y, z, push);
}
DispatchCommand Program::dispatch(uint32_t x, uint32_t y, uint32_t z) const {
    return dispatch({}, x, y, z);
}

DispatchIndirectCommand Program::dispatchIndirect(
    std::span<const std::byte> push, const Tensor<std::byte>& tensor, uint64_t offset) const
{
    return DispatchIndirectCommand(*program, tensor, offset, push);
}
DispatchIndirectCommand Program::dispatchIndirect(const Tensor<std::byte>& tensor, uint64_t offset) const {
    return DispatchIndirectCommand(*program, tensor, offset, {});
}

Program::Program(Program&&) noexcept = default;
Program& Program::operator=(Program&&) noexcept = default;

Program::Program(ContextHandle context, std::span<const uint32_t> code, std::span<const std::byte> specialization)
    : Resource(std::move(context))
    , program(std::make_unique<vulkan::Program>(*getContext()))
{
    //context reference
    auto& con = getContext();

    //create reflection module
    SpvReflectShaderModule reflectModule;
    spvCheckResult(spvReflectCreateShaderModule2(
        SPV_REFLECT_MODULE_FLAG_NO_COPY,
        code.size_bytes(), code.data(),
        &reflectModule));

    //save local size
    //we assume that there is exactly one entry_point
    program->localSize = {
        .x = reflectModule.entry_points->local_size.x,
        .y = reflectModule.entry_points->local_size.y,
        .z = reflectModule.entry_points->local_size.z
    };

    //create pipeline layout; we fill this as we gather more info
    VkPipelineLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    };

    //We only support a single descriptor set
    if (reflectModule.descriptor_set_count > 1)
        throw std::logic_error("Programs are only allowed to have a single descriptor set!");
    //build descriptor set layout
    if (reflectModule.descriptor_set_count == 1) {
        auto& set = reflectModule.descriptor_sets[0];

        //count actually used params
        auto pBinding = *set.bindings;
        auto pBindingEnd = pBinding + set.binding_count;
        auto param_count = static_cast<uint32_t>(std::count_if(pBinding, pBindingEnd,
            [](const SpvReflectDescriptorBinding& b) -> bool { return b.accessed != 0; }
        ));

        //reserve binding slots
        program->boundParams = std::vector<VkWriteDescriptorSet>(param_count);
        bindingTraits.resize(param_count);

        //Create bindings
        std::unordered_set<uint32_t> bindingSet;
        std::vector<VkDescriptorSetLayoutBinding> bindings(param_count);
        for (auto i = -1; pBinding != pBindingEnd; ++pBinding) {
            //if the file was compiled with auto binding mapping, unused bindings get mapped to 0
            //since this might result in multiple bindigs pointing to the same, overwriting each
            //other, we HAVE to skip unused ones. While this will later produce errors if the
            //user wants to bind an unused parameter, he still can check the bindings of the
            //program to see which bindings he can actually use
            if (!pBinding->accessed)
                continue;
            else
                ++i;

            //save binding number to set so we can later check if there were any duplicates
            if (bindingSet.contains(pBinding->binding))
                throw std::runtime_error("Invalid shader code: A binding number has been assigned to multiple inputs!");
            else
                bindingSet.insert(pBinding->binding);

            if (pBinding->count == 0)
                throw std::runtime_error("Unbound arrays are not supported!");

            bindings[i] = VkDescriptorSetLayoutBinding{
                .binding         = pBinding->binding,
                .descriptorType  = static_cast<VkDescriptorType>(pBinding->descriptor_type),
                .descriptorCount = pBinding->count,
                .stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT
            };
            program->boundParams[i] = VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstBinding = pBinding->binding,
                .descriptorCount = pBinding->count,
                .descriptorType = static_cast<VkDescriptorType>(pBinding->descriptor_type)
            };

            bindingTraits[i] = {
                .name = pBinding->name ? pBinding->name : "",
                .binding = pBinding->binding,
                .type = static_cast<ParameterType>(pBinding->descriptor_type),
                .count = pBinding->count
            };

            switch (pBinding->descriptor_type) {
            case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_IMAGE:
                bindingTraits[i].imageTraits = ImageBindingTraits{
                    .format = castImageFormat(pBinding->image.image_format),
                    .dims = castDimension(pBinding->image.dim)
                };
                break;
            case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER:
            case SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
                auto name = pBinding->type_description->type_name;
                bindingTraits[i].name = name ? name : "";
                break;
            }
        }

        //Create set layout
        VkDescriptorSetLayoutCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
            .bindingCount = param_count,
            .pBindings    = bindings.data()
        };
        vulkan::checkResult(con->fnTable.vkCreateDescriptorSetLayout(
            con->device, &info, nullptr, &program->descriptorSetLayout));

        //register in pipeline layout
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &program->descriptorSetLayout;
    }

    //read push descriptor
    VkPushConstantRange push{};
    if (reflectModule.push_constant_block_count > 1)
        throw std::logic_error("Multiple push constant found, but only up to one is supported!");
    if (reflectModule.push_constant_block_count == 1) {
        //read push size
        push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push.size = reflectModule.push_constant_blocks->size;

        //register in layout
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &push;
    }

    //Create pipeline layout
    vulkan::checkResult(con->fnTable.vkCreatePipelineLayout(
        con->device, &layoutInfo, nullptr, &program->pipeLayout));

    //read specialization constants map
    auto totalSpecSlots = reflectModule.spec_constant_count;
    auto specSlots = std::min(totalSpecSlots,
        static_cast<uint32_t>(specialization.size_bytes() / 4));
    std::vector<VkSpecializationMapEntry> specMap(specSlots);
    if (specSlots > 0) {
        //reflection might have ids out of order
        // -> first fetch all ids, than sort them
        std::vector<uint32_t> specIds(totalSpecSlots);
        for (auto i = 0u; i < totalSpecSlots; ++i) {
            specIds[i] = reflectModule.spec_constants[i].constant_id;
        }
        std::sort(specIds.begin(), specIds.end());
        //create map entries only up to certain amount to match provided data
        //(remaining ones use default values)
        for (auto i = 0u; i < specSlots; ++i) {
            //all types allowed for specialization are 4 bytes long
            //we assume the spec const to be tightly packed
            specMap[i] = VkSpecializationMapEntry{
                .constantID = specIds[i],
                .offset = 4 * i,
                .size = 4
            };
        }
    }

    //we're done reflecting
    spvReflectDestroyShaderModule(&reflectModule);

    //build specialization info
    VkSpecializationInfo specInfo{
        .mapEntryCount = specSlots,
        .pMapEntries   = specMap.data(),
        .dataSize      = specialization.size_bytes(),
        .pData         = specialization.data()
    };

    //compile code into shader module
    VkShaderModuleCreateInfo shaderInfo{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = code.size_bytes(),
        .pCode = code.data()
    };
    vulkan::checkResult(con->fnTable.vkCreateShaderModule(
        con->device, &shaderInfo, nullptr, &program->shader));

    //Shader stage create info
    VkPipelineShaderStageCreateInfo stageInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = program->shader,
            .pName = reflectModule.entry_point_name,
            .pSpecializationInfo = specMap.empty() ? nullptr : &specInfo
    };

    //create compute pipeline
    VkComputePipelineCreateInfo pipeInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = stageInfo,
        .layout = program->pipeLayout,
    };
    vulkan::checkResult(con->fnTable.vkCreateComputePipelines(
        con->device, con->cache,
        1, &pipeInfo,
        nullptr, &program->pipeline));
}
Program::Program(ContextHandle context, std::span<const uint32_t> code)
    : Program(std::move(context), code, {})
{}
Program::~Program() {
    auto& context = getContext();
    if (program) {
        context->fnTable.vkDestroyPipeline(context->device, program->pipeline, nullptr);
        context->fnTable.vkDestroyShaderModule(context->device, program->shader, nullptr);
        context->fnTable.vkDestroyPipelineLayout(context->device, program->pipeLayout, nullptr);
        context->fnTable.vkDestroyDescriptorSetLayout(context->device, program->descriptorSetLayout, nullptr);
    }
}

/********************************* FLUSH MEMORY *******************************/

namespace {

const VkMemoryBarrier memoryBarrier {
    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
    .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
    .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT
};

}

void FlushMemoryCommand::record(vulkan::Command& cmd) const {
    cmd.stage |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    context.get().fnTable.vkCmdPipelineBarrier(cmd.buffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_DEPENDENCY_BY_REGION_BIT,
        1, &memoryBarrier,
        0, nullptr,
        0, nullptr);
}

FlushMemoryCommand::FlushMemoryCommand(const FlushMemoryCommand&) = default;
FlushMemoryCommand& FlushMemoryCommand::operator=(const FlushMemoryCommand&) = default;

FlushMemoryCommand::FlushMemoryCommand(const ContextHandle& context)
    : context(*context)
{}
FlushMemoryCommand::~FlushMemoryCommand() = default;

}
