#pragma once

#include <optional>
#include <span>

#include "hephaistos/argument.hpp"
#include "hephaistos/context.hpp"
#include "hephaistos/handles.hpp"

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
	uint64_t vertexStride = 3 * sizeof(float);
	uint32_t vertexCount = 0;
	std::span<const uint32_t> indices = {};
};

struct GeometryInstance {
	std::reference_wrapper<const Geometry> geometry;
	TransformMatrix transform = IdentityTransform;
	uint32_t customIndex : 24 = 0;
	uint32_t mask : 8         = 0xFF;
};

class HEPHAISTOS_API AccelerationStructure : public Argument, public Resource {
public:
	void bindParameter(VkWriteDescriptorSet& binding) const override final;

	AccelerationStructure(const AccelerationStructure&) = delete;
	AccelerationStructure& operator=(const AccelerationStructure&) = delete;

	AccelerationStructure(AccelerationStructure&&) noexcept;
	AccelerationStructure& operator=(AccelerationStructure&&) noexcept;

	AccelerationStructure(ContextHandle context, const GeometryInstance& instance);
	AccelerationStructure(ContextHandle context, std::span<const GeometryInstance> instances);
	~AccelerationStructure() override;

private:
	struct Parameter;
	std::unique_ptr<Parameter> param;
};

//template<>
//class HEPHAISTOS_API ArgumentArray<AccelerationStructure> : public Argument {
//public:
//	using ElementType = AccelerationStructure;
//
//public:
//	void bindParameter(VkWriteDescriptorSet& binding) const override final;
//
//	[[nodiscard]] size_t size() const;
//	[[nodiscard]] bool empty() const;
//
//	void clear();
//	void resize(size_t count);
//
//	void set(size_t pos, const ElementType& img);
//	void push_back(const ElementType& img);
//
//	ArgumentArray(const ArgumentArray& other);
//	ArgumentArray& operator=(const ArgumentArray& other);
//
//
//	ArgumentArray(ArgumentArray&& other) noexcept;
//	ArgumentArray& operator=(ArgumentArray&& other) noexcept;
//
//	ArgumentArray();
//	ArgumentArray(size_t count);
//	ArgumentArray(size_t count, const ElementType& img);
//	ArgumentArray(std::initializer_list<std::reference_wrapper<const ElementType>> images);
//
//	~ArgumentArray() override;
//
//private:
//	struct Argument;
//	std::unique_ptr<Argument> argument;
//};

}
