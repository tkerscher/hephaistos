#include "hephaistos/atomic.hpp"

#include <algorithm>
#include <array>

#include "volk.h"

#include "vk/types.hpp"

namespace hephaistos {

namespace {

constexpr auto ExtensionName = "Atomics";

//reinterpret_cast struct to bool* should be fine in most case
//but is UB nonetheless. Hence this wonky construct.
constexpr uint32_t toBitFlags(const AtomicsProperties& props) {
    uint32_t v = 0;

    v += props.bufferInt64Atomics << 0;

    v += props.bufferFloat16Atomics << 1;
    v += props.bufferFloat16AtomicAdd << 2;
    v += props.bufferFloat16AtomicMinMax << 3;

    v += props.bufferFloat32Atomics << 4;
    v += props.bufferFloat32AtomicAdd << 5;
    v += props.bufferFloat32AtomicMinMax << 6;

    v += props.bufferFloat64Atomics << 7;
    v += props.bufferFloat64AtomicAdd << 8;
    v += props.bufferFloat64AtomicMinMax << 9;

    v += props.sharedInt64Atomics << 10;

    v += props.sharedFloat16Atomics << 11;
    v += props.sharedFloat16AtomicAdd << 12;
    v += props.sharedFloat16AtomicMinMax << 13;

    v += props.sharedFloat32Atomics << 14;
    v += props.sharedFloat32AtomicAdd << 15;
    v += props.sharedFloat32AtomicMinMax << 16;

    v += props.sharedFloat64Atomics << 17;
    v += props.sharedFloat64AtomicAdd << 18;
    v += props.sharedFloat64AtomicMinMax << 19;

    v += props.imageInt64Atomics << 20;

    v += props.imageFloat32Atomics << 21;
    v += props.imageFloat32AtomicAdd << 22;
    v += props.imageFloat32AtomicMinMax << 23;

    return v;
}

//flags to determine which extensions we'll need
constexpr auto imageExtFlags = toBitFlags({
    .imageInt64Atomics = true
});
constexpr auto int64ExtFlags = toBitFlags({
    .bufferInt64Atomics = true,
    .sharedInt64Atomics = true
});
constexpr auto floatExt1Flags = toBitFlags({
    .bufferFloat32Atomics = true,
    .bufferFloat32AtomicAdd = true,
    .bufferFloat64Atomics = true,
    .bufferFloat64AtomicAdd = true,
    .sharedFloat32Atomics = true,
    .sharedFloat32AtomicAdd = true,
    .sharedFloat64Atomics = true,
    .sharedFloat64AtomicAdd = true,
    .imageFloat32Atomics = true,
    .imageFloat32AtomicAdd = true
});
constexpr auto floatExt2Flags = toBitFlags({
    .bufferFloat16Atomics = true,
    .bufferFloat16AtomicAdd = true,
    .bufferFloat16AtomicMinMax = true,
    .bufferFloat32AtomicMinMax = true,
    .bufferFloat64AtomicMinMax = true,
    .sharedFloat16Atomics = true,
    .sharedFloat16AtomicAdd = true,
    .sharedFloat16AtomicMinMax = true,
    .sharedFloat32AtomicMinMax = true,
    .sharedFloat64AtomicMinMax = true,
    .imageFloat32AtomicMinMax = true
});

constexpr AtomicsProperties NoAtomics{};

}

class AtomicsExtension : public Extension {
public:
    const AtomicsProperties props;

    bool isDeviceSupported(const DeviceHandle& device) const override {
        auto supported = toBitFlags(getAtomicsProperties(device));
        return (supported & flags) == flags;
    }
    std::string_view getExtensionName() const override {
        return ExtensionName;
    }
    std::span<const char* const> getDeviceExtensions() const override {
        return extensions;
    }
    void* chain(void* pNext) override {
        int64Feat.pNext = pNext;
        return this->pNext;
    }

    AtomicsExtension(const AtomicsProperties& props);
    virtual ~AtomicsExtension() = default;

private:
    uint32_t flags;

    std::vector<const char*> extensions;
    void* pNext;

    VkPhysicalDeviceShaderImageAtomicInt64FeaturesEXT imageFeat;
    VkPhysicalDeviceShaderAtomicInt64FeaturesKHR int64Feat;
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT floatFeat;
    VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT float2Feat;
};

AtomicsProperties getAtomicsProperties(const DeviceHandle& device) {
    //nullcheck
    if (!device)
        return {};

    //We dont check for extension support as we might not need
    //all of them

    //query features
    VkPhysicalDeviceShaderImageAtomicInt64FeaturesEXT feat4{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_IMAGE_ATOMIC_INT64_FEATURES_EXT,
    };
    VkPhysicalDeviceShaderAtomicInt64FeaturesKHR feat3{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES_KHR,
        .pNext = &feat4
    };
    VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT feat2{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_2_FEATURES_EXT,
        .pNext = &feat3
    };
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT feat1{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT,
        .pNext = &feat2
    };
    VkPhysicalDeviceFeatures2 feat{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &feat1
    };
    vkGetPhysicalDeviceFeatures2(device->device, &feat);

    //copy features
    //double negate to make the compiler happy (implicit narrowing)...
    return {
        !!feat3.shaderBufferInt64Atomics,
        !!feat2.shaderBufferFloat16Atomics,
        !!feat2.shaderBufferFloat16AtomicAdd,
        !!feat2.shaderBufferFloat16AtomicMinMax,
        !!feat1.shaderBufferFloat32Atomics,
        !!feat1.shaderBufferFloat32AtomicAdd,
        !!feat2.shaderBufferFloat32AtomicMinMax,
        !!feat1.shaderBufferFloat64Atomics,
        !!feat1.shaderBufferFloat64AtomicAdd,
        !!feat2.shaderBufferFloat64AtomicMinMax,
        !!feat3.shaderSharedInt64Atomics,
        !!feat2.shaderSharedFloat16Atomics,
        !!feat2.shaderSharedFloat16AtomicAdd,
        !!feat2.shaderSharedFloat16AtomicMinMax,
        !!feat1.shaderSharedFloat32Atomics,
        !!feat1.shaderSharedFloat32AtomicAdd,
        !!feat2.shaderSharedFloat32AtomicMinMax,
        !!feat1.shaderSharedFloat64Atomics,
        !!feat1.shaderSharedFloat64AtomicAdd,
        !!feat2.shaderSharedFloat64AtomicMinMax,
        !!feat4.shaderImageInt64Atomics,
        !!feat1.shaderImageFloat32Atomics,
        !!feat1.shaderImageFloat32AtomicAdd,
        !!feat2.shaderImageFloat32AtomicMinMax
    };    
}

const AtomicsProperties& getEnabledAtomics(const ContextHandle& context) {
    auto& ext = context->extensions;
    auto pExt = std::find_if(ext.begin(), ext.end(),
        [](const ExtensionHandle& h) -> bool {
            return h->getExtensionName() == ExtensionName;
        });
    if (pExt == ext.end())
        return NoAtomics;
    else
        return dynamic_cast<AtomicsExtension*>(pExt->get())->props;
}

AtomicsExtension::AtomicsExtension(const AtomicsProperties& props)
    : props(props)
    , flags(toBitFlags(props))
    , extensions({})
{
    //check which extensions we need
    bool imageExt = (flags & imageExtFlags) != 0;
    bool floatExt = (flags & floatExt1Flags) != 0;
    bool float2Ext = (flags & floatExt2Flags) != 0;
    if (imageExt)
        extensions.push_back(VK_EXT_SHADER_IMAGE_ATOMIC_INT64_EXTENSION_NAME);
    if (floatExt)
        extensions.push_back(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
    if (float2Ext)
        extensions.push_back(VK_EXT_SHADER_ATOMIC_FLOAT_2_EXTENSION_NAME);
    //VK_KHR_SHADER_ATOMIC_INT64 is part of Vulkan 1.2

    //fill feature structs
    int64Feat = VkPhysicalDeviceShaderAtomicInt64FeaturesKHR{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES_KHR,
        .shaderBufferInt64Atomics = props.bufferInt64Atomics,
        .shaderSharedInt64Atomics = props.sharedInt64Atomics
    };
    pNext = &int64Feat;
    imageFeat = VkPhysicalDeviceShaderImageAtomicInt64FeaturesEXT{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_IMAGE_ATOMIC_INT64_FEATURES_EXT,
        .pNext = pNext,
        .shaderImageInt64Atomics = props.imageInt64Atomics
    };
    if (imageExt)
        pNext = static_cast<void*>(&imageFeat);
    floatFeat = VkPhysicalDeviceShaderAtomicFloatFeaturesEXT{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT,
        .pNext = pNext,
        .shaderBufferFloat32Atomics = props.bufferFloat32Atomics,
        .shaderBufferFloat32AtomicAdd = props.bufferFloat32AtomicAdd,
        .shaderBufferFloat64Atomics = props.bufferFloat64Atomics,
        .shaderBufferFloat64AtomicAdd = props.bufferFloat64AtomicAdd,
        .shaderSharedFloat32Atomics = props.sharedFloat32Atomics,
        .shaderSharedFloat32AtomicAdd = props.sharedFloat32AtomicAdd,
        .shaderSharedFloat64Atomics = props.sharedFloat64Atomics,
        .shaderSharedFloat64AtomicAdd = props.sharedFloat64AtomicAdd,
        .shaderImageFloat32Atomics = props.imageFloat32Atomics,
        .shaderImageFloat32AtomicAdd = props.imageFloat32AtomicAdd
    };
    if (floatExt)
        pNext = static_cast<void*>(&floatFeat);
    float2Feat = VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_2_FEATURES_EXT,
        .pNext = pNext,
        .shaderBufferFloat16Atomics = props.bufferFloat16Atomics,
        .shaderBufferFloat16AtomicAdd = props.bufferFloat16AtomicAdd,
        .shaderBufferFloat16AtomicMinMax = props.bufferFloat16AtomicMinMax,
        .shaderBufferFloat32AtomicMinMax = props.bufferFloat32AtomicMinMax,
        .shaderBufferFloat64AtomicMinMax = props.bufferFloat64AtomicMinMax,
        .shaderSharedFloat16Atomics = props.sharedFloat16Atomics,
        .shaderSharedFloat16AtomicAdd = props.sharedFloat16AtomicAdd,
        .shaderSharedFloat16AtomicMinMax = props.sharedFloat16AtomicMinMax,
        .shaderSharedFloat32AtomicMinMax = props.sharedFloat32AtomicMinMax,
        .shaderSharedFloat64AtomicMinMax = props.sharedFloat64AtomicMinMax,
        .shaderImageFloat32AtomicMinMax = props.imageFloat32AtomicMinMax
    };
    if (float2Ext)
        pNext = static_cast<void*>(&float2Feat);
}
ExtensionHandle createAtomicsExtension(const AtomicsProperties& properties) {
    return std::make_unique<AtomicsExtension>(properties);
}
}
