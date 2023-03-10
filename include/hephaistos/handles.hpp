#pragma once

#include <memory>

namespace hephaistos {

//fwd
namespace vulkan {
    struct Buffer;
    struct Context;
    struct Device;
}

using ContextHandle = std::shared_ptr<vulkan::Context>;

using BufferHandle = std::unique_ptr<vulkan::Buffer, void(*)(vulkan::Buffer*)>;
using DeviceHandle = std::unique_ptr<vulkan::Device, void(*)(vulkan::Device*)>;

}
