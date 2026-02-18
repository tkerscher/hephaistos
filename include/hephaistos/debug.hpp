#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <type_traits>
#include <vector>

#include "hephaistos/config.hpp"
#include "hephaistos/handles.hpp"

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

/**
 * @brief Enumeration of address fault types that may have caused a device loss
 */
enum class DeviceFaultAddressType {
	None = 0,
	ReadInvalid = 1,
	WriteInvalid = 2,
	ExecuteInvalid = 3,
	InstructionPointerUnknown = 4,
	InstructionPointerInvalid = 5,
	InstructionPointerFault = 6
};

/**
 * @brief Specifies memory address at which a fault occurred
 */
struct DeviceFaultAddressInfo {
	/**
	 * @brief Type of memory operation that triggered the fault
	 */
	DeviceFaultAddressType addressType;
	/**
	 * @brief Address at which the fault occurred
	 */
	uint64_t address;
	/**
	 * @brief Precision of the reported address
	 */
	uint64_t precision;
};

/**
 * @brief Vendor specific fault information
 */
struct DeviceFaultVendorInfo {
	/**
	 * @brief Human readable description of the fault
	 */
	std::string description;
	/**
	 * @brief Vendor specific fault code
	 */
	uint64_t code;
	/**
	 * @brief Vendor specific data associated with fault
	 */
	uint64_t data;
};

/**
 * @brief Structure containing information about device fault
 */
struct DeviceFaultInfo {
	/**
	 * @brief Human readable description of fault
	 */
	std::string description;
	/**
	 * @brief List of address faults
	 */
	std::vector<DeviceFaultAddressInfo> addressInfo;
	/**
	 * @brief List of vendor specific faults
	 */
	std::vector<DeviceFaultVendorInfo> vendorInfo;
};

/**
 * @brief Queries whether the given device supports device fault information.
 */
[[nodiscard]] HEPHAISTOS_API
bool isDeviceFaultExtensionSupported(const DeviceHandle& device);

/**
 * @brief Creates extension enabling device fault information.
 */
[[nodiscard]] HEPHAISTOS_API ExtensionHandle createDeviceFaultInfoExtension();

/**
 * @brief Queries information about the last device lost error.
 * 
 * Can be called after a device lost error occurred to gather information about
 * its cause. May only be called after such an error occurred.
 */
[[nodiscard]] HEPHAISTOS_API
DeviceFaultInfo getDeviceFaultInfo(const ContextHandle& context);

}
