#pragma once

#include <memory>

namespace hephaistos {

//fwd
namespace vulkan {
    struct Buffer;
    struct Context;
    struct Device;
    struct Image;
    struct Timeline;
    struct Param;
}

using ContextHandle = std::shared_ptr<vulkan::Context>;

using BufferHandle   = std::unique_ptr<vulkan::Buffer,   void(*)(vulkan::Buffer*)>;
using DeviceHandle   = std::unique_ptr<vulkan::Device,   void(*)(vulkan::Device*)>;
using ImageHandle    = std::unique_ptr<vulkan::Image, void(*)(vulkan::Image*)>;
using TimelineHandle = std::unique_ptr<vulkan::Timeline, void(*)(vulkan::Timeline*)>;

struct Extension;

using ExtensionHandle = std::unique_ptr<Extension>;

}
