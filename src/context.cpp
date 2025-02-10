#include "hephaistos/context.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>

#include "hephaistos/handles.hpp"
#include "vk/instance.hpp"
#include "vk/result.hpp"
#include "vk/types.hpp"

namespace hephaistos {

bool isVulkanAvailable() {
    //Check if volk is initialized or try to do so
    return (volkGetInstanceVersion() != 0) || (volkInitialize() == VK_SUCCESS);
}

/************************************ DEVICE *********************************/

namespace {

constexpr VkQueueFlags QueueFlags = VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
constexpr float QueuePriority = 1.0f;

constexpr auto DeviceExtensions = std::to_array({
    VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
    //only needed for gpu printf, but widely supported
    VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME
});

void destroyDevice(vulkan::Device* device) {
    delete device;
    vulkan::returnInstance();
}

DeviceHandle createDevice(VkPhysicalDevice device) {
    //increment instance ref count
    vulkan::getInstance();

    //query extensions
    uint32_t count;
    vulkan::checkResult(vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr));
    std::vector<VkExtensionProperties> extensions(count);
    vulkan::checkResult(vkEnumerateDeviceExtensionProperties(device, nullptr, &count, extensions.data()));

    //allocate handle
    DeviceHandle result{
        new vulkan::Device{ device, std::vector<std::string>(count) },
        destroyDevice
    };

    //copy extensions
    std::transform(extensions.begin(), extensions.end(), result->supportedExtensions.begin(),
        [](VkExtensionProperties prop) -> std::string { return std::string(prop.extensionName); });
    //sort extensions for std::includes
    std::sort(result->supportedExtensions.begin(), result->supportedExtensions.end());

    //done
    return result;
};

DeviceInfo createInfo(VkPhysicalDevice device) {
    //retrieve props
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(device, &props);

    //return and build info
    return {
        .name       = props.deviceName,
        .isDiscrete = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
    };
}

}

bool isDeviceSuitable(const DeviceHandle& device, std::span<const ExtensionHandle> extensions) {
    //Check for the following things:
    //    - Compute queue
    //    - feature support
    //  - Extension support

    //Check for features support
    {
        VkPhysicalDeviceVulkan12Features features12{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES
        };
        VkPhysicalDeviceFeatures2 features{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = &features12
        };
        vkGetPhysicalDeviceFeatures2(device->device, &features);
        //timeline sempahore feature
        if (!features12.timelineSemaphore)
            return false;
        //buffer device address
        if (!features12.bufferDeviceAddress)
            return false;
        //host query reset (stopwatch)
        if (!features12.hostQueryReset)
            return false;
        //scalar block layout
        if (!features12.scalarBlockLayout)
            return false;
    }

    //check for compute queue
    {
        uint32_t count;
        vkGetPhysicalDeviceQueueFamilyProperties(device->device, &count, nullptr);
        std::vector<VkQueueFamilyProperties> props(count);
        vkGetPhysicalDeviceQueueFamilyProperties(device->device, &count, props.data());
        bool found = false;
        for (auto& queue : props) {
            if ((queue.queueFlags & QueueFlags) == QueueFlags) {
                found = true;
                break;
            }
        }
        if (!found)
            return false;
    }

    //Check for internal extension support
    if (!DeviceExtensions.empty() && !std::includes(
        device->supportedExtensions.begin(),
        device->supportedExtensions.end(),
        DeviceExtensions.begin(),
        DeviceExtensions.end()))
    {
        return false;
    }

    //check for external extension support
    for (auto& ext : extensions) {
        if (!ext->isDeviceSupported(device))
            return false;
    }

    //Everything checked
    return true;
}

std::vector<DeviceHandle> enumerateDevices() {
    //get instance
    auto instance = vulkan::getInstance();

    //get all devices
    uint32_t count;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance, &count, devices.data());

    //transform supported devices
    std::vector<DeviceHandle> result;
    for (auto device : devices) {
        result.push_back(createDevice(device));
    }

    //we only needed the instance for the enumerate command
    //device handles have their own refs so return it
    vulkan::returnInstance();

    //done
    return result;
}

DeviceHandle getDevice(const ContextHandle& context) {
    return createDevice(context->physicalDevice);
}

DeviceInfo getDeviceInfo(const DeviceHandle& device) {
    return createInfo(device->device);
}

DeviceInfo getDeviceInfo(const ContextHandle& context) {
    return createInfo(context->physicalDevice);
}

/*********************************** CONTEXT *********************************/

namespace {

void destroyContext(vulkan::Context* context) {
    if (context->allocator)
        vmaDestroyAllocator(context->allocator);
    context->fnTable.vkDestroyPipelineCache(context->device, context->cache, nullptr);
    context->fnTable.vkDestroyFence(context->device, context->oneTimeSubmitFence, nullptr);
    context->fnTable.vkDestroyCommandPool(context->device, context->oneTimeSubmitPool, nullptr);
    context->fnTable.vkDestroyCommandPool(context->device, context->subroutinePool, nullptr);
    while (!context->sequencePool.empty()) {
        context->fnTable.vkDestroyCommandPool(
            context->device, context->sequencePool.front(), nullptr);
        context->sequencePool.pop();
    }
    context->fnTable.vkDestroyDevice(context->device, nullptr);
    vulkan::returnInstance();
    delete context;
}

ContextHandle createContext(
    VkInstance instance,
    VkPhysicalDevice device,
    std::span<ExtensionHandle> extensions)
{
    //Allocate memory
    ContextHandle context{ new vulkan::Context, destroyContext };
    context->physicalDevice = device;

    //query queue family
    uint32_t family = 0;
    {
        uint32_t count;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
        std::vector<VkQueueFamilyProperties> props(count);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &count, props.data());
        for (; family < count; ++family) {
            if (props[family].queueFlags & QueueFlags)
                break;
        }
    }
    context->queueFamily = family;

    //Create logical device
    {
        //check for certain shader subgroup features
        VkPhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR controlFlow{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_UNIFORM_CONTROL_FLOW_FEATURES_KHR
        };
        VkPhysicalDeviceShaderMaximalReconvergenceFeaturesKHR reconvergence{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MAXIMAL_RECONVERGENCE_FEATURES_KHR,
            .pNext = &controlFlow
        };
        //Check for extended arithmetic type support (e.f. float64)
        //and enable them by default
        //(since we can't reasonable chain basic feature set)
        VkPhysicalDeviceVulkan12Features features12{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
            .pNext = &reconvergence
        };
        VkPhysicalDeviceFeatures2 features2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = &features12
        };
        vkGetPhysicalDeviceFeatures2(device, &features2);

        //extensions are allowed to add device extensions, but
        //vulkan wants their names in contiguous memory
        // -> collect them in a vector
        //TODO: Right now it's not possible, but we should
        //        check to not name an extension multiple times
        //Note: Passing the same extension multiple times is
        //        the users fault
        std::vector<const char*> allDeviceExtensions{};
        //copy the base one
        allDeviceExtensions.insert(
            allDeviceExtensions.end(),
            DeviceExtensions.begin(),
            DeviceExtensions.end());

        //process extensions
        void* pNext = nullptr;
        for (auto& ext : extensions) {
            //chain features
            pNext = ext->chain(pNext);
            //add extensions
            auto extNames = ext->getDeviceExtensions();
            allDeviceExtensions.insert(
                allDeviceExtensions.end(),
                extNames.begin(), extNames.end());
            //save extension
            context->extensions.emplace_back(std::move(ext));
        }

        //chain optional features if available
        if (controlFlow.shaderSubgroupUniformControlFlow) {
            controlFlow.pNext = pNext;
            pNext = static_cast<void*>(&controlFlow);
            allDeviceExtensions.push_back(
                VK_KHR_SHADER_SUBGROUP_UNIFORM_CONTROL_FLOW_EXTENSION_NAME);
        }
        if (reconvergence.shaderMaximalReconvergence) {
            reconvergence.pNext = pNext;
            pNext = static_cast<void*>(&reconvergence);
            allDeviceExtensions.push_back(
                VK_KHR_SHADER_MAXIMAL_RECONVERGENCE_EXTENSION_NAME);
        }
        //obligatory features
        VkDeviceQueueCreateInfo queueInfo{
            .sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = family,
            .queueCount       = 1,
            .pQueuePriorities = &QueuePriority
        };
        VkPhysicalDeviceTimelineSemaphoreFeatures timeline{
            .sType             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
            .pNext             = pNext, //chain external extensions
            .timelineSemaphore = VK_TRUE
        };
        VkPhysicalDeviceHostQueryResetFeatures hostQueryReset{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES,
            .pNext = &timeline,
            .hostQueryReset = VK_TRUE
        };
        VkPhysicalDeviceShaderFloat16Int8Features float16Int8Features{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
            .pNext = &hostQueryReset,
            .shaderFloat16 = features12.shaderFloat16,
            .shaderInt8    = features12.shaderInt8,
        };
        VkPhysicalDeviceScalarBlockLayoutFeatures scalarBlockFeatures{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES,
            .pNext = &float16Int8Features,
            .scalarBlockLayout = VK_TRUE
        };
        VkPhysicalDeviceBufferDeviceAddressFeatures addressFeatures{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
            .pNext = &scalarBlockFeatures,
            .bufferDeviceAddress = VK_TRUE
        };
        VkPhysicalDeviceFeatures features{
            .shaderFloat64 = features2.features.shaderFloat64,
            .shaderInt64   = features2.features.shaderInt64,
            .shaderInt16   = features2.features.shaderInt16
        };
        VkDeviceCreateInfo deviceInfo{
            .sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pNext                   = &addressFeatures,
            .queueCreateInfoCount    = 1,
            .pQueueCreateInfos       = &queueInfo,
            .enabledExtensionCount   = static_cast<uint32_t>(allDeviceExtensions.size()),
            .ppEnabledExtensionNames = allDeviceExtensions.data(),
            .pEnabledFeatures        = &features
        };
        vulkan::checkResult(vkCreateDevice(device, &deviceInfo, nullptr, &context->device));
    }

    //load device functions
    volkLoadDeviceTable(&context->fnTable, context->device);
    //get queue
    context->fnTable.vkGetDeviceQueue(context->device, family, 0, &context->queue);

    //create command pool for subroutines
    {
        VkCommandPoolCreateInfo poolInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .queueFamilyIndex = family
        };
        vulkan::checkResult(context->fnTable.vkCreateCommandPool(
            context->device, &poolInfo, nullptr, &context->subroutinePool));
    }
    //create command pool for internal one time submits
    {
        VkCommandPoolCreateInfo poolInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
            .queueFamilyIndex = family
        };
        vulkan::checkResult(context->fnTable.vkCreateCommandPool(
            context->device, &poolInfo, nullptr, &context->oneTimeSubmitPool));
    }
    //create command buffer for one time submits
    {
        VkCommandBufferAllocateInfo bufInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = context->oneTimeSubmitPool,
            .commandBufferCount = 1
        };
        vulkan::checkResult(context->fnTable.vkAllocateCommandBuffers(
            context->device, &bufInfo, &context->oneTimeSubmitBuffer));
    }
    //create fence for synchronizing internal one time submits
    {
        VkFenceCreateInfo fenceInfo{
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO
        };
        vulkan::checkResult(context->fnTable.vkCreateFence(
            context->device, &fenceInfo, nullptr, &context->oneTimeSubmitFence));
    }

    //Create pipeline cache
    {
        VkPipelineCacheCreateInfo cacheInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO
        };
        vulkan::checkResult(context->fnTable.vkCreatePipelineCache(
            context->device, &cacheInfo, nullptr, &context->cache));
    }

    //Create allocator
    {
        VmaVulkanFunctions functions{
            .vkGetPhysicalDeviceProperties           = vkGetPhysicalDeviceProperties,
            .vkGetPhysicalDeviceMemoryProperties     = vkGetPhysicalDeviceMemoryProperties,
            .vkAllocateMemory                        = context->fnTable.vkAllocateMemory,
            .vkFreeMemory                            = context->fnTable.vkFreeMemory,
            .vkMapMemory                             = context->fnTable.vkMapMemory,
            .vkUnmapMemory                           = context->fnTable.vkUnmapMemory,
            .vkFlushMappedMemoryRanges               = context->fnTable.vkFlushMappedMemoryRanges,
            .vkInvalidateMappedMemoryRanges          = context->fnTable.vkInvalidateMappedMemoryRanges,
            .vkBindBufferMemory                      = context->fnTable.vkBindBufferMemory,
            .vkBindImageMemory                       = context->fnTable.vkBindImageMemory,
            .vkGetBufferMemoryRequirements           = context->fnTable.vkGetBufferMemoryRequirements,
            .vkGetImageMemoryRequirements            = context->fnTable.vkGetImageMemoryRequirements,
            .vkCreateBuffer                          = context->fnTable.vkCreateBuffer,
            .vkDestroyBuffer                         = context->fnTable.vkDestroyBuffer,
            .vkCreateImage                           = context->fnTable.vkCreateImage,
            .vkDestroyImage                          = context->fnTable.vkDestroyImage,
            .vkCmdCopyBuffer                         = context->fnTable.vkCmdCopyBuffer,
            .vkGetBufferMemoryRequirements2KHR       = context->fnTable.vkGetBufferMemoryRequirements2,
            .vkGetImageMemoryRequirements2KHR        = context->fnTable.vkGetImageMemoryRequirements2,
            .vkBindBufferMemory2KHR                  = context->fnTable.vkBindBufferMemory2,
            .vkBindImageMemory2KHR                   = context->fnTable.vkBindImageMemory2,
            .vkGetPhysicalDeviceMemoryProperties2KHR = vkGetPhysicalDeviceMemoryProperties2,
            .vkGetDeviceBufferMemoryRequirements     = context->fnTable.vkGetDeviceBufferMemoryRequirementsKHR,
            .vkGetDeviceImageMemoryRequirements      = context->fnTable.vkGetDeviceImageMemoryRequirementsKHR
        };

        VmaAllocatorCreateInfo info{
            .flags            = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
            .physicalDevice   = device,
            .device           = context->device,
            .pVulkanFunctions = &functions,
            .instance         = instance,
            .vulkanApiVersion = VK_API_VERSION_1_2
        };

        vulkan::checkResult(vmaCreateAllocator(&info, &context->allocator));
    }

    //Done
    return context;
}

}

ContextHandle createContext(std::span<ExtensionHandle> extensions) {
    //new handle -> refs to instance
    auto instance = vulkan::getInstance();

    //enumerate devices
    uint32_t count;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance, &count, devices.data());

    //select first suitable discrete GPU
    VkPhysicalDevice fallback = nullptr;
    VkPhysicalDeviceProperties props;
    for (auto device : devices) {
        if (isDeviceSuitable(createDevice(device), extensions)) {
            //Remember last suitable device if no discrete present
            fallback = device;
            vkGetPhysicalDeviceProperties(device, &props);
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
                return createContext(instance, device, extensions);
        }
    }

    //no discrete gpu
    if (fallback)
        return createContext(instance, fallback, extensions);
    else
        throw std::runtime_error("No suitable device available!");
}

ContextHandle createContext(
    const DeviceHandle& device,
    std::span<ExtensionHandle> extensions)
{
    if (!device)
        throw std::runtime_error("Device is empty!");
    if (!isDeviceSuitable(device, extensions))
        throw std::runtime_error("Device is not suitable!");

    //new ref
    return createContext(vulkan::getInstance(), device->device, extensions);
}

/*********************************** RESOURCE ********************************/

Resource::operator bool() const noexcept {
    return bool(context);
}

void Resource::destroy() {
    if (!context) return;

    //destroy resource
    onDestroy();

#ifdef HEPHAISTOS_MANAGED_RESOURCES
    //remove this resource from list
    //skip this step if we modify in bulk
    if (!context->resources_locked) {
        std::erase(context->resources, this);
    }
#endif

    //return context pointer
    context.reset();
}

const ContextHandle& Resource::getContext() const noexcept {
    return context;
}

#ifdef HEPHAISTOS_MANAGED_RESOURCES
Resource::Resource(Resource&& other) noexcept
    : context(std::move(other.context)) 
{
    //update pointer in resource list
    if (context) {
        auto it = std::find(
            context->resources.begin(),
            context->resources.end(),
            &other
        );
        if (it != context->resources.end())
            *it = this;
    }
}
Resource& Resource::operator=(Resource&& other) {
    //destroy old resource if present
    if (context) destroy();

    context = std::move(other.context);
    //update pointer in resource list
    if (context) {
        auto it = std::find(
            context->resources.begin(),
            context->resources.end(),
            &other
        );
        if (it != context->resources.end())
            *it = this;
    }
    return *this;
}

Resource::~Resource() {
    //deregister resource if not yet destroyed
    if (*this) {
        context->resources.erase(std::remove(
            context->resources.begin(),
            context->resources.end(),
            this
        ), context->resources.end());
    }
}
#else
Resource::Resource(Resource&& other) noexcept = default;
Resource& Resource::operator=(Resource&& other) {
    //destroy old resource if present
    if (context) destroy();
    //move context
    context = std::move(other.context);
}

Resource::~Resource() = default;
#endif

Resource::Resource(ContextHandle context)
    : context(std::move(context))
{
#ifdef HEPHAISTOS_MANAGED_RESOURCES
    //register resource
    this->context->resources.push_back(this);
#endif
}

#ifdef HEPHAISTOS_MANAGED_RESOURCES

/******************************* RESOURCE SNAPSHOT ***************************/

size_t ResourceSnapshot::count() const noexcept {
    if (context.expired()) return 0;

    auto ptr = context.lock();
    return std::count_if(ptr->resources.begin(), ptr->resources.end(),
        [&snapshot = snapshot](Resource* res) {
            return res && *res && !snapshot.contains(res);
        }
    );
}

void ResourceSnapshot::capture() {
    if (context.expired())
        throw std::runtime_error("Can not capture resources. Context have been destroyed!");

    auto ptr = context.lock();
    snapshot = std::unordered_set<Resource*>(
        ptr->resources.begin(), ptr->resources.end()
    );
}

void ResourceSnapshot::restore() {
    if (context.expired()) return;

    auto ptr = context.lock();
    //mark following operation as bulk
    ptr->resources_locked = true;
    //destroy and remove resources in reverse order
    std::erase_if(ptr->resources,
        [&snapshot = snapshot](Resource* res) -> bool {
            bool remove = !snapshot.contains(res);
            if (res && remove) res->destroy();
            return remove;
        });
    //unlock
    ptr->resources_locked = false;
}

ResourceSnapshot::ResourceSnapshot(const ContextHandle& context)
    : context(context)
{ }

size_t getResourceCount(const ContextHandle& context) noexcept {
    return context ? context->resources.size() : 0;
}

void destroyAllResources(const ContextHandle& context) {
    if (!context) return;

    //set bulk mode
    context->resources_locked = true;
    //destroy all resources
    for (auto res : context->resources)
        res->destroy();
    context->resources.clear();
    //finish bulk mode
    context->resources_locked = false;

    //take the opportunity to clean up context
    //-> free up memory taken by memory pools
    context->fnTable.vkTrimCommandPool(context->device, context->oneTimeSubmitPool, 0);
    context->fnTable.vkTrimCommandPool(context->device, context->subroutinePool, 0);
    while (!context->sequencePool.empty()) {
        context->fnTable.vkDestroyCommandPool(
            context->device, context->sequencePool.front(), nullptr);
        context->sequencePool.pop();
    }
}

#endif

}
