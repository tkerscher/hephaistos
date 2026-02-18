#include <nanobind/nanobind.h>

#include <hephaistos/exception.hpp>

namespace hp = hephaistos;
namespace nb = nanobind;
using namespace nb::literals;

void registerExceptionModule(nb::module_& m) {
    auto vulkan_error = nb::exception<hp::vulkan_error>(m, "VulkanError", PyExc_RuntimeError);
    nb::exception<hp::out_of_device_memory_error>(m, "OutOfDeviceMemoryError", vulkan_error);
    nb::exception<hp::device_lost_error>(m, "DeviceLostError", vulkan_error);
}
