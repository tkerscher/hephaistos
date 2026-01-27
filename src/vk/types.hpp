#pragma once

#include <algorithm>
#include <queue>
#include <string>
#include <string_view>
#include <vector>

#include "volk.h"
#include "vk_mem_alloc.h"
#include "hephaistos/config.hpp"
#include "hephaistos/context.hpp"
#include "hephaistos/handles.hpp"

namespace hephaistos::vulkan {

struct Buffer {
    VkBuffer buffer;
    VmaAllocation allocation;
    VmaAllocationInfo allocInfo;

    const Context& context;
};

struct Command {
    VkCommandBuffer buffer;
    //specifies which stage the commands used so the semaphores
    //can be more fine grained.
    VkPipelineStageFlags stage;

    //const Context& context;
};

struct Context {
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkQueue queue;
    VkPipelineCache cache;

    //List of enabled hephaistos extensions (not vulkan!)
    std::vector<ExtensionHandle> extensions;

    uint32_t queueFamily;
    VkCommandPool subroutinePool;
    VkCommandPool oneTimeSubmitPool;
    VkCommandBuffer oneTimeSubmitBuffer;
    VkFence oneTimeSubmitFence;

    //for synchronization operations like the ones needed for copy
    //commands we also want to include ray tracing operations.
    //However, the spec only allows us to include that stage if
    //ray tracing pipelines have been enabled. Since we do not want
    //to cluster the code with checks for ray tracing support, we
    //instead have save the correct stage flags here
    VkPipelineStageFlags computeStages =
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

    //looks like a hack, but this is the only one we need to change
    //would be a bit drastic to remove const Context because of this
    mutable std::queue<VkCommandPool> sequencePool;

    VmaAllocator allocator;

    VolkDeviceTable fnTable;

#ifdef HEPHAISTOS_MANAGED_RESOURCES
    std::vector<Resource*> resources;
    //set to true, if resources vector should not be modifed
    //used on bulk destroying to prevent iterators from becoming invalid
    //and for better performance
    bool resources_locked = false;
#endif
};

struct Device {
    VkPhysicalDevice device;
    std::vector<std::string> supportedExtensions;
};

struct Image {
    VkImage image;
    VkImageView view;
    VmaAllocation allocation;

    const Context& context;
};

struct Timeline {
    VkSemaphore semaphore;
};

template<class Extension>
const Extension* getExtension(const Context& context, const char* name) {
    auto pExt = std::find_if(
        context.extensions.begin(),
        context.extensions.end(),
        [name](const ExtensionHandle& h) -> bool {
            return h->getExtensionName() == name;
        }
    );
    if (pExt == context.extensions.end())
        return nullptr;
    else
        return dynamic_cast<const Extension*>(pExt->get());
}

[[nodiscard]] BufferHandle createBuffer(
    const ContextHandle& handle,
    uint64_t size,
    VkBufferUsageFlags usage,
    VmaAllocationCreateFlags flags);
[[nodiscard]] BufferHandle createBufferAligned(
    const ContextHandle& context,
    uint64_t size,
    uint64_t alignment,
    VkBufferUsageFlags usage,
    VmaAllocationCreateFlags flags);
void destroyBuffer(Buffer* buffer);
[[nodiscard]] inline BufferHandle createEmptyBuffer() {
    return { nullptr, destroyBuffer };
}

[[nodiscard]] VkDeviceAddress getBufferDeviceAddress(const BufferHandle& buffer);

[[nodiscard]] ImageHandle createImage(
    const ContextHandle& context,
    VkFormat format,
    uint32_t width, uint32_t height, uint32_t depth,
    VkImageUsageFlags usage);
void destroyImage(Image* image);

}
