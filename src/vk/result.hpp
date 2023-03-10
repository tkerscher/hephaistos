#pragma once

#include <stdexcept>
#include "volk.h"

#define STR(c) case VK_##c: return #c;

namespace hephaistos::vulkan {

inline auto printResult(VkResult res) {
    switch (res) {
        STR(SUCCESS)
        STR(NOT_READY)
        STR(TIMEOUT)
        STR(EVENT_SET)
        STR(EVENT_RESET)
        STR(INCOMPLETE)
        STR(ERROR_OUT_OF_HOST_MEMORY)
        STR(ERROR_OUT_OF_DEVICE_MEMORY)
        STR(ERROR_INITIALIZATION_FAILED)
        STR(ERROR_DEVICE_LOST)
        STR(ERROR_MEMORY_MAP_FAILED)
        STR(ERROR_LAYER_NOT_PRESENT)
        STR(ERROR_EXTENSION_NOT_PRESENT)
        STR(ERROR_FEATURE_NOT_PRESENT)
        STR(ERROR_INCOMPATIBLE_DRIVER)
        STR(ERROR_TOO_MANY_OBJECTS)
        STR(ERROR_FORMAT_NOT_SUPPORTED)
        STR(ERROR_FRAGMENTED_POOL)
        STR(ERROR_OUT_OF_POOL_MEMORY)
        STR(ERROR_INVALID_EXTERNAL_HANDLE)
        STR(ERROR_SURFACE_LOST_KHR)
        STR(ERROR_NATIVE_WINDOW_IN_USE_KHR)
        STR(SUBOPTIMAL_KHR)
        STR(ERROR_OUT_OF_DATE_KHR)
        STR(ERROR_INCOMPATIBLE_DISPLAY_KHR)
        STR(ERROR_VALIDATION_FAILED_EXT)
        STR(ERROR_INVALID_SHADER_NV)
        STR(ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT)
        STR(ERROR_FRAGMENTATION_EXT)
        STR(ERROR_NOT_PERMITTED_EXT)
        STR(ERROR_INVALID_DEVICE_ADDRESS_EXT)
        STR(ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT)
        STR(RESULT_MAX_ENUM)
    default:
        return "Unknown result code";
    }
}

#undef STR

inline void checkResult(VkResult res) {
    if (res != VK_SUCCESS) {
        throw std::runtime_error(printResult(res));
    }
}

}