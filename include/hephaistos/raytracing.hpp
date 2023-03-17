#pragma once

#include <optional>
#include <span>

#include "hephaistos/context.hpp"
#include "hephaistos/handles.hpp"

struct VkWriteDescriptorSet;

namespace hephaistos {

[[nodiscard]] HEPHAISTOS_API bool isRaytracingSupported(const DeviceHandle& device);
[[nodiscard]] HEPHAISTOS_API bool isRaytracingEnabled(const ContextHandle& context);

[[nodiscard]] HEPHAISTOS_API ExtensionHandle createRaytracingExtension();

struct TransformMatrix {
	float matrix[3][4];
};
constexpr TransformMatrix IdentityTransform = {
	1.0f, 0.0f, 0.0f, 0.0f,
	0.0f, 1.0f, 0.0f, 0.0f,
	0.0f, 0.0f, 1.0f, 0.0f
};

struct Geometry {
	std::span<const std::byte> vertices = {};
	uint64_t vertexStride = 0;
	uint32_t vertexCount = 0;
	std::span<const uint32_t> indices = {};
};

struct GeometryInstance {
	std::reference_wrapper<const Geometry> geometry;
	TransformMatrix transform = IdentityTransform;
	uint32_t customIndex : 24 = 0;
	uint32_t mask : 8         = 0xFF;
};

class HEPHAISTOS_API AccelerationStructure : public Resource {
public:
	void bindParameter(VkWriteDescriptorSet& binding) const;

	AccelerationStructure(const AccelerationStructure&) = delete;
	AccelerationStructure& operator=(const AccelerationStructure&) = delete;

	AccelerationStructure(AccelerationStructure&&) noexcept;
	AccelerationStructure& operator=(AccelerationStructure&&) noexcept;

	AccelerationStructure(ContextHandle context, const GeometryInstance& instance);
	AccelerationStructure(ContextHandle context, std::span<const GeometryInstance> instances);
	virtual ~AccelerationStructure();

private:
	struct Parameter;
	std::unique_ptr<Parameter> param;
};

}
