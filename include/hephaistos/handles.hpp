#pragma once

#include <memory>

namespace hephaistos {

//fwd
namespace vulkan {
    struct Context;
    struct Device;
}

using ContextHandle = std::shared_ptr<vulkan::Context>;

using DeviceHandle = std::unique_ptr<vulkan::Device, void(*)(vulkan::Device*)>;

}
