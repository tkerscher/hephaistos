#include "vk/reflection.hpp"

#include <sstream>

#include "spirv_reflect.h"

#include "vk/result.hpp"

namespace hephaistos {

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

VkShaderStageFlagBits castShaderStage(SpvReflectShaderStageFlagBits stage) {
    //these use the same numeric values
    return static_cast<VkShaderStageFlagBits>(stage);
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

BindingTraits getTraits(SpvReflectDescriptorBinding* pBinding) {
    BindingTraits trait{
                .name = pBinding->name ? pBinding->name : "",
                .binding = pBinding->binding,
                .type = static_cast<ParameterType>(pBinding->descriptor_type),
                .count = pBinding->count
    };

    switch (pBinding->descriptor_type) {
    case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_IMAGE:
        trait.imageTraits = ImageBindingTraits{
            .format = castImageFormat(pBinding->image.image_format),
            .dims = castDimension(pBinding->image.dim)
        };
        break;
    case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER:
    case SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
        auto name = pBinding->type_description->type_name;
        trait.name = name ? name : "";
        break;
    }

    return trait;
}

bool addTraitChecked(std::vector<BindingTraits>& traits, const BindingTraits& trait) {
    //see if this binding is already in use
    auto binding = trait.binding;
    auto it = std::find_if(
        traits.begin(), traits.end(),
        [binding](const BindingTraits& t) -> bool {
            return t.binding == binding;
        });
    //new binding
    if (it == traits.end()) {
        traits.push_back(trait);
        return true;
    }
    //binding already in use -> check if it's the same
    if (*it != trait) {
        std::stringstream stream;
        stream << "Duplicate binding " << binding;
        if (!trait.name.empty())
            stream << " (" << trait.name << ") ";
        stream << "does not match previous definition!";
        throw std::runtime_error(stream.str());
    }
    return false;
}

}

namespace vulkan {

void LayoutReflectionBuilder::add(std::span<const uint32_t> code) {
	//create reflection module
	SpvReflectShaderModule reflectModule;
    spvCheckResult(spvReflectCreateShaderModule2(
        SPV_REFLECT_MODULE_FLAG_NO_COPY,
        code.size_bytes(),
        code.data(),
        &reflectModule));

    //save local size
    //we assume that there is exactly one entry_point
    localSize = {
        reflectModule.entry_points->local_size.x,
        reflectModule.entry_points->local_size.y,
        reflectModule.entry_points->local_size.z
    };
    //fetch which stage we are on
    auto stage = castShaderStage(reflectModule.entry_points->shader_stage);

    //We only support a single descriptor set
    if (reflectModule.descriptor_set_count > 1)
        throw std::logic_error("Programs are only allowed to have a single descriptor set!");
    //update descriptor set layout
    if (reflectModule.descriptor_set_count == 1) {
        auto& set = reflectModule.descriptor_sets[0];

        //count actually used params
        auto pBinding = *set.bindings;
        auto pBindingEnd = pBinding + set.binding_count;
        auto param_count = static_cast<uint32_t>(std::count_if(pBinding, pBindingEnd,
            [](const SpvReflectDescriptorBinding& b) -> bool { return b.accessed != 0; }
        ));

        //reserve binding slots
        bindings.reserve(param_count);
        params.reserve(param_count);
        traits.reserve(param_count);

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

            if (pBinding->count == 0)
                throw std::runtime_error("Unbound arrays are not supported!");

            //check if binding is already present, and if so, if it's the same
            if (addTraitChecked(traits, getTraits(pBinding))) {
                //new one -> add rest
                bindings.push_back(VkDescriptorSetLayoutBinding{
                    .binding = pBinding->binding,
                    .descriptorType = static_cast<VkDescriptorType>(pBinding->descriptor_type),
                    .descriptorCount = pBinding->count,
                    .stageFlags = VK_SHADER_STAGE_ALL //TODO: Update stage flags each time ?
                });
                params.push_back(VkWriteDescriptorSet{
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstBinding = pBinding->binding,
                    .descriptorCount = pBinding->count,
                    .descriptorType = static_cast<VkDescriptorType>(pBinding->descriptor_type)
                });
            }
        } // for each binding
    } // if descriptor set

    //update push constant range
    if (reflectModule.push_constant_block_count > 1)
        throw std::runtime_error("Multiple push constant found, but only up to one is supported!");
    if (reflectModule.push_constant_block_count == 1) {
        push.stageFlags |= stage;
        push.size = std::max(push.size, reflectModule.push_constant_blocks->size);
    }

    //update specialization map
    for (auto i = 0; i < reflectModule.spec_constant_count; ++i) {
        specializationIds.insert(reflectModule.spec_constants[i].constant_id);
    }

    //we're done reflecting
    spvReflectDestroyShaderModule(&reflectModule);
}

void LayoutReflectionBuilder::createDescriptorSetLayout(const Context& context, VkDescriptorSetLayout* layout) {
    VkDescriptorSetLayoutCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings = bindings.data()
    };
    vulkan::checkResult(context.fnTable.vkCreateDescriptorSetLayout(
        context.device, &info, nullptr, layout));
}

std::vector<VkSpecializationMapEntry> LayoutReflectionBuilder::createSpecializationMap(size_t count) {
    //collect and sort all found ids
    std::vector<uint32_t> ids(specializationIds.begin(), specializationIds.end());
    std::sort(ids.begin(), ids.end());
    //create map; only use the first count ones. The rest will use default values
    auto n = std::min(ids.size(), count);
    std::vector<VkSpecializationMapEntry> map(n);
    for (auto i = 0u; i < n; ++i) {
        //all types allowed for specialization are 4 bytes long
        //we assume the spec const to be tightly packed
        map[i] = VkSpecializationMapEntry{
            .constantID = ids[i],
            .offset = 4 * i,
            .size = 4
        };
    }

    return map;
}

void LayoutReflectionBuilder::createPipelineLayout(
    const Context& context,
    const VkDescriptorSetLayout* pDescriptorSetLayout,
    VkPipelineLayout* pPipelineLayout
) {
    VkPipelineLayoutCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = pDescriptorSetLayout ? 1u : 0u,
        .pSetLayouts = pDescriptorSetLayout,
        .pushConstantRangeCount = push.size > 0 ? 1u : 0u,
        .pPushConstantRanges = push.size > 0 ? &push : nullptr
    };
    vulkan::checkResult(context.fnTable.vkCreatePipelineLayout(
        context.device, &info, nullptr, pPipelineLayout));
}

LayoutReflectionBuilder::LayoutReflectionBuilder()
	: params()
	, bindings()
	, traits()
	, specializationIds()
	, push({})
{}

}

}
