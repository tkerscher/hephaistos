#pragma once

#include <stdexcept>

#include "hephaistos/config.hpp"

namespace hephaistos {

/**
 * @brief Generic error raised by the Vulkan API
 */
HEPHAISTOS_API class vulkan_error : public std::runtime_error{
public:
	explicit vulkan_error(const char* msg);
	~vulkan_error() override;
};

/**
 * @brief Allocation failed due to insufficient device memory
 */
HEPHAISTOS_API class out_of_device_memory_error : public vulkan_error {
public:
	explicit out_of_device_memory_error();
	~out_of_device_memory_error() override;
};

/**
 * @brief Device lost error
 * 
 * Raised when the device encounters an unrecoverable error.
 * Any data belonging to the device is undefined and further operations
 * are not allowed.
 */
HEPHAISTOS_API class device_lost_error : public vulkan_error {
public:
	explicit device_lost_error();
	~device_lost_error() override;
};

}
