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

struct Mesh {
    std::span<const std::byte> vertices = {};
    uint32_t vertexStride               = 3 * sizeof(float);
    std::span<const uint32_t> indices   = {};
};

struct Geometry {
    uint64_t blas_address     = 0;
    uint64_t vertices_address = 0;
    uint64_t indices_address  = 0;
};

struct GeometryInstance {
    uint64_t blas_address     = 0;
    TransformMatrix transform = IdentityTransform;
    uint32_t customIndex : 24 = 0;
    uint32_t mask        : 8  = 0xFF;
};

class HEPHAISTOS_API GeometryStore : public Resource {
public:
    [[nodiscard]]
    const std::vector<Geometry>& geometries() const noexcept;
    [[nodiscard]]
    const Geometry& operator[](size_t idx) const;

    [[nodiscard]] size_t size() const noexcept;

    [[nodiscard]]
    GeometryInstance createInstance(
        size_t idx,
        const TransformMatrix& transform = IdentityTransform,
        uint32_t customIndex = 0,
        uint8_t mask = 0xFF) const;

    GeometryStore(const GeometryStore&) = delete;
    GeometryStore& operator=(const GeometryStore&) = delete;

    GeometryStore(GeometryStore&&) noexcept;
    GeometryStore& operator=(GeometryStore&&) noexcept;

    GeometryStore(
        ContextHandle context,
        const Mesh& mesh,
        bool keepMeshData = false);
    GeometryStore(
        ContextHandle context,
        std::span<const Mesh> meshes,
        bool keepMeshData = false);
    ~GeometryStore() override;

private:
    struct Imp;
    std::unique_ptr<Imp> pImp;
};

class HEPHAISTOS_API AccelerationStructure : public Argument, public Resource {
public:
    void bindParameter(VkWriteDescriptorSet& binding) const final override;

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

}
