#pragma once

#include <memory>

namespace hephaistos {

//fwd
namespace vulkan {
    struct Buffer;
    struct Command;
    struct Context;
    struct Device;
    struct Timeline;
}

using ContextHandle = std::shared_ptr<vulkan::Context>;

using BufferHandle   = std::unique_ptr<vulkan::Buffer,   void(*)(vulkan::Buffer*)>;
using CommandHandle  = std::unique_ptr<vulkan::Command,  void(*)(vulkan::Command*)>;
using DeviceHandle   = std::unique_ptr<vulkan::Device,   void(*)(vulkan::Device*)>;
using TimelineHandle = std::unique_ptr<vulkan::Timeline, void(*)(vulkan::Timeline*)>;

}
