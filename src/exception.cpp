#include "hephaistos/exception.hpp"

namespace hephaistos {

vulkan_error::vulkan_error(const char* msg)
    : std::runtime_error(msg)
{}
vulkan_error::~vulkan_error() = default;

out_of_device_memory_error::out_of_device_memory_error()
    : vulkan_error("Out of device memory")
{}
out_of_device_memory_error::~out_of_device_memory_error() = default;

device_lost_error::device_lost_error()
    : vulkan_error("Device lost")
{}
device_lost_error::~device_lost_error() = default;

}
