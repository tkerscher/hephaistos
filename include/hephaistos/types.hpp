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

}
