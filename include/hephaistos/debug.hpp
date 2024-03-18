#pragma once

#include <cstdint>
#include <functional>
#include <type_traits>

#include "hephaistos/config.hpp"

namespace hephaistos {

/**
 * @brief Flags indicating debug message severity
*/
enum class DebugMessageSeverityFlagBits : int32_t {
	VERBOSE_BIT = 0x0001,
	INFO_BIT    = 0x0010,
	WARNING_BIT = 0x0100,
	ERROR_BIT   = 0x1000,
};

inline int32_t operator|(
	DebugMessageSeverityFlagBits lhs, DebugMessageSeverityFlagBits rhs
) {
	return static_cast<int32_t>(lhs) | static_cast<int32_t>(rhs);
}

inline int32_t operator&(
	DebugMessageSeverityFlagBits lhs, DebugMessageSeverityFlagBits rhs
) {
	return static_cast<int32_t>(lhs) & static_cast<int32_t>(rhs);
}

/**
 * @brief Single debug message
*/
struct DebugMessage {
	/**
	 * @brief Severity of this message
	*/
	DebugMessageSeverityFlagBits severity;
	/**
	 * Message name as defined by the validation layer
	*/
	const char* pIdName;
	/**
	 * @brief Message ID used by the validation layer
	*/
	int32_t idNumber;
	/**
	 * @brief null-terminated c-string containing the message text
	*/
	const char* pMessage;
};

using DebugCallback = std::function<void(const DebugMessage&)>;

/**
 * Options for configuring the debug state
*/
struct DebugOptions {
	/***
	 * @brief Enables the usage of GL_EXT_debug_printf
	*/
	bool enablePrint = false;
	/**
	 * @brief Enable GPU assisted validation
	*/
	bool enableGPUValidation = false;
	/**
	 * @brief Enable synchronization validation between resources
	*/
	bool enableSynchronizationValidation = false;
	/**
	 * @brief Enable thread safety validation
	*/
	bool enableThreadSafetyValidation = false;

	/**
	 * @brief Enables validation of the Vulkan API usage
	*/
	bool enableAPIValidation = false;
};

/**
 * @brief Checks whether debugging features are available
 * 
 * @note Debugging relies on the Vulkan Validation Layers being installed
*/
[[nodiscard]] HEPHAISTOS_API bool isDebugAvailable();

/**
 * @brief Configures the debug state
 * 
 * @param options Options used to configure the debug state
 * @param callback Callback functions called on each message.
 *                 Prints to cout by default.
 * 
 * @note This only takes effect if called before any other function
 *       except isDebugAvailable() and isVulkanAvailable()
*/
HEPHAISTOS_API void configureDebug(
	DebugOptions options,
	DebugCallback callback = nullptr);

}
