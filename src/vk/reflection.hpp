#pragma once

#include <span>
#include <vector>

#include "volk.h"

#include "hephaistos/bindings.hpp"

#include "vk/types.hpp"

namespace hephaistos::vulkan {

class LayoutReflectionBuilder {
public:
    std::vector<VkWriteDescriptorSet> params;
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    std::vector<BindingTraits> traits;
    //std::vector<VkSpecializationMapEntry> specializationMap;
    std::unordered_set<uint32_t> specializationIds;

    VkPushConstantRange push;

    struct LocalSize {
        uint32_t x, y, z;
    } localSize;

    void add(std::span<const uint32_t> code);

    void createDescriptorSetLayout(const Context& context, VkDescriptorSetLayout* layout);
    std::vector<VkSpecializationMapEntry> createSpecializationMap(size_t count);
    void createPipelineLayout(
        const Context& context,
        const VkDescriptorSetLayout* pDescriptorSetLayout,
        VkPipelineLayout* pPipelineLayout);

    LayoutReflectionBuilder();
};

inline VkSpecializationInfo createSpecializationInfo(
    std::span<const VkSpecializationMapEntry> specMap,
    std::span<const std::byte> data    
) {
    return VkSpecializationInfo{
        .mapEntryCount = static_cast<uint32_t>(specMap.size()),
        .pMapEntries = specMap.data(),
        .dataSize = data.size_bytes(),
        .pData = data.data()
    };
}

}
