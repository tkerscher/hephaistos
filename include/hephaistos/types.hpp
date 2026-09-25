#pragma once

#include "hephaistos/context.hpp"

namespace hephaistos {

/**
 * @brief List of optional type support in programs
 */
struct TypeSupport {
	bool float64;
	bool float16;

	bool int64;
	bool int16;
	bool int8;

	bool buffer16BitAccess;
	bool uniform16BitAccess;
	bool pushConstant16BitAccess;

	bool buffer8BitAccess;
	bool uniform8BitAccess;
	bool pushConstant8BitAccess;
};

/**
 * @brief Returns the extended types the device supports
 * 
 * @note All supported types are enabled by default
 * @param device Device to query
*/
[[nodiscard]]
HEPHAISTOS_API TypeSupport getSupportedTypes(const DeviceHandle& device);
/**
 * @brief Returns the extended types the context supports
 * 
 * @param context Context to query
*/
[[nodiscard]] TypeSupport getSupportedTypes(const ContextHandle& context);

/**
 * @brief Creates an extension to make certain types required
 * 
 * Creates an extension that marks the given types as required and thus
 * will discard devices, that do not support them. Usefull for automatic
 * device selection
 * 
 * @note Extended types are enabled regardless of this extension
*/
[[nodiscard]]
HEPHAISTOS_API ExtensionHandle createTypeExtension(const TypeSupport& types);

/**
 * @brief List of types for which the device supports OpFmaKHR
 * 
 * Vulkan does not guarantee that the fused-multiply-add (fma) intrinsic is
 * a fused operation and thus code cannot rely on the improved accuracy.
 * The SPIR-V intrinsic OpFmaKHR gives that guarantee, but its support is
 * optional and depends on the floating point type.
 */
struct FmaSupport {
	bool float16;
	bool float32;
	bool float64;
};

/**
 * @brief Returns the types for which the device supports OpFmaKHR
 * 
 * @note All supported types are enabled by default
 * @param device Device to query
 */
[[nodiscard]]
HEPHAISTOS_API FmaSupport getFmaSupport(const DeviceHandle& device);
/**
 * @brief Returns the types for which the context supports OpFmaKHR
 * 
 * @param context Context to query
 */
[[nodiscard]]
HEPHAISTOS_API FmaSupport getFmaSupport(const ContextHandle& context);

/**
 * @brief Creates an extension to make certain OpFmaKHR support required
 * 
 * Creates an extension that enforces the support of OpFmaKHR for the given
 * types. Devices not meeting that requirement will thus be discarded.
 * 
 * @note Supported types are enabled regardless of this extension
 */
[[nodiscard]]
HEPHAISTOS_API ExtensionHandle createFmaExtension(const FmaSupport& fma);

}
