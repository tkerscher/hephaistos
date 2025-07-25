#pragma once

#include <optional>
#include <span>

#include "hephaistos/argument.hpp"
#include "hephaistos/command.hpp"
#include "hephaistos/context.hpp"
#include "hephaistos/handles.hpp"

namespace hephaistos {

/**
 * @brief Checks for ray tracing support
 * 
 * @param device Handle to device to be checked for ray tracing support
 * @return True, if the given device supports ray tracing, false otherwise.
*/
[[nodiscard]] HEPHAISTOS_API bool isRaytracingSupported(const DeviceHandle& device);
/**
 * @brief Checks wether raytracing is enabled
 * 
 * @param context Context to check
 * @return True, if ray tracing is enabled in the given context, false otherwise.
*/
[[nodiscard]] HEPHAISTOS_API bool isRaytracingEnabled(const ContextHandle& context);

/**
 * @brief Creates a ray tracing extension
 * 
 * Returns an extension which can be passed during the creation of a context to
 * enable ray tracing.
 * 
 * @return Extension for enabling ray tracing
*/
[[nodiscard]] HEPHAISTOS_API ExtensionHandle createRaytracingExtension();

/**
 * @brief Transformation matrix
 * 
 * Transformation matrix applied to geometries to create instances: \n
 *      / x' \   / m00 m01 m02 m03 \   / x \ \n
 *      | y' | = | m10 m11 m12 m13 | * | y | \n
 *      | z' |   | m20 m21 m22 m23 |   | z | \n
 *      \ 1  /   \  0   0   0   1  /   \ 1 / \n
 * 
 * @see GeometryInstance
*/
struct TransformMatrix {
    /**
     * @brief 3x4 row major transformation matrix
    */
    float matrix[3][4];
};
/**
 * @brief Identity transformation matrix
*/
constexpr TransformMatrix IdentityTransform = {
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f
};

/**
 * @brief Polygon mesh
 * 
 * Mesh consisting of triangles used to create geometries used in ray tracing.
 * Allows for arbitrary vertex format via a stride as well as indexing of
 * vertices.
 * 
 * @see Geometry, GeometryStore
*/
struct Mesh {
    /**
     * @brief Memory range containing the vertex data
    */
    std::span<const std::byte> vertices = {};
    /**
     * @brief Stride between vertices in the vertex data
    */
    uint32_t vertexStride               = 3 * sizeof(float);
    /**
     * @brief Memory range containing the index data. May be empty.
    */
    std::span<const uint32_t> indices   = {};
};

/**
 * @brief Geometry represents a Mesh prepared for ray tracing
 * 
 * Geometry contains device memory addresses to the stored BLAS, an opaque
 * device specific representation of a Mesh used to build acceleration
 * structures, as well as the uploaded vertex and index data.
 * 
 * @note Vertex and index data are always uploaded to the device for processing
 *       but may be discarded afterwards to save memory space. In this case the
 *       corresponding addresses are zero.
 * 
 * @see Mesh, GeometryStore
*/
struct Geometry {
    /**
     * @brief Device memory address containing the BLAS
    */
    uint64_t blas_address     = 0;
    /**
     * @brief Device memory address containing the vertex data
    */
    uint64_t vertices_address = 0;
    /**
     * @brief Device memory address containing the index data
    */
    uint64_t indices_address  = 0;
};

/**
 * @brief Instance referencing a Geometry
 * 
 * Instance references a Geometry and applies additional information such as
 * transformation and mask.
 * 
 * @see AccelerationStructure, Geometry, GeometryStore, TransformationMatrix
*/
struct GeometryInstance {
    /**
     * @brief Device memory address of BLAS. Zero marks this instance as inactive.
    */
    uint64_t blas_address     = 0;
    /**
     * @brief Transformation applied to the referenced Geometry
    */
    TransformMatrix transform = IdentityTransform;
    /**
     * @brief Custom index retrievable during ray tracing
    */
    uint32_t customIndex : 24 = 0;
    /**
     * @brief Mask to conditionally enable geometry in ray tracing.
    */
    uint32_t mask        : 8  = 0xFF;
};

/**
 * @brief Factory class for creating GeometryInstance from a set of Mesh
 * 
 * Upon construction GeometryStore creates a Geometry for each provided Mesh and
 * provides methods for creating GeometryInstance from them.
 * 
 * @see Geometry, GeometryInstance, Mesh
*/
class HEPHAISTOS_API GeometryStore : public Resource {
public:
    /**
     * @brief Returns the vector containing every Geometry inside this store
    */
    [[nodiscard]] const std::vector<Geometry>& geometries() const noexcept;
    /**
     * @brief Returns the i-th Geometry
     * 
     * @note order is the same as the one of the Meshes passed to the constructor
     * @param idx Index of the Geometry to return
    */
    [[nodiscard]] const Geometry& operator[](size_t idx) const;

    /**
     * @brief Returns the amount of Geometry in this store
    */
    [[nodiscard]] size_t size() const noexcept;

    /**
     * @brief Creates a GeometryInstance referencing the i-th Geometry
     * 
     * @param idx Index of the Geometry to reference
     * @param transform Transformation matrix to apply to the instance
     * @param customIndex Custom index of the instance
     * @param mask Mask of the instance
     * 
     * @see GeometryInstance, TransformationMatrix
    */
    [[nodiscard]]
    GeometryInstance createInstance(
        size_t idx,
        const TransformMatrix& transform = IdentityTransform,
        uint32_t customIndex = 0,
        uint8_t mask = 0xFF) const;

    GeometryStore(const GeometryStore&) = delete;
    GeometryStore& operator=(const GeometryStore&) = delete;

    GeometryStore(GeometryStore&&) noexcept;
    GeometryStore& operator=(GeometryStore&&);

    /**
     * @brief Creates a new GeometryStore
     * 
     * @note Device memory addresses to the mesh data are stored inside each
     *       Geometry
     * 
     * @param context Context on which to create the GeometryStore
     * @param mesh Mesh to create a Geometry from
     * @param keepMeshData If False, deletes mesh data from device memory after
     *                     creating Geometry.
     * 
     * @see Geometry, Mesh
    */
    GeometryStore(
        ContextHandle context,
        const Mesh& mesh,
        bool keepMeshData = false);
    /**
     * @brief Creates a new GeometryStore
     * 
     * Creates a Geometry for each Mesh provided. Geometries can later be
     * referenced using their index which matches the index of the mesh provided
     * to create it.
     * 
     * @note Device memory addresses to the mesh data are stored inside each
     *       Geometry
     * 
     * @param context Context on which to create the GeometryStore
     * @param meshes List of Mesh to create Geometry from
     * @param keepMeshData If False, deletes mesh data from device memory after
     *                     creating Geometry.
     * 
     * @see Geometry, Mesh
    */
    GeometryStore(
        ContextHandle context,
        std::span<const Mesh> meshes,
        bool keepMeshData = false);
    ~GeometryStore() override;

protected:
    void onDestroy() override;

private:
    struct Imp;
    std::unique_ptr<Imp> pImp;
};

/**
 * @brief Acceleration structure used for tracing rays against
 * 
 * Acceleration structure are built from a set of GeometryInstance and can be
 * bound to a Program to trace ray against.
 * 
 * @see GeometryInstance, GeometryStore
*/
class HEPHAISTOS_API AccelerationStructure : public Argument, public Resource {
public:
    void bindParameter(VkWriteDescriptorSet& binding) const final override;

    /**
     * @brief Returns the number of instances that can fit in the acceleration structure
    */
    [[nodiscard]] uint32_t capacity() const noexcept;
    /**
     * @brief Returns the current amount of instances in the acceleration structure
     */
    [[nodiscard]] uint32_t size() const noexcept;

    /**
     * @brief Device address containing the instance data
     * 
     * Returns the device buffer address of where the contiguous array of
     * VkAccelerationStructureInstanceKHR elements used to build the
     * acceleration structure is stored. Will throw if the acceleration
     * structure is frozen.
    */
    [[nodiscard]] uint64_t instanceBufferAddress() const;
    /**
     * @brief Uses given instances to update acceleration structure.
     * 
     * Updates the acceleration structure using the given instances. Will throw
     * if the instances exceed the acceleration structure's capacity or if it is
     * frozen. Unused instances in the acceleration structure will become inactive.
     */
    void update(std::span<const GeometryInstance> instances);

    /**
     * @brief Whether the acceleration structure has been frozen.
     * 
     * Returns true, if the acceleration structure is frozen and cannot be
     * altered, or false otherwise.
     */
    [[nodiscard]] bool frozen() const noexcept;
    void freeze();

    AccelerationStructure(const AccelerationStructure&) = delete;
    AccelerationStructure& operator=(const AccelerationStructure&) = delete;

    AccelerationStructure(AccelerationStructure&&) noexcept;
    AccelerationStructure& operator=(AccelerationStructure&&);

    /**
     * @brief Creates a new empty acceleration structure to be filled later
     * 
     * @param context Context on which to create the AccelerationStructure
     * @param capacity Maximum amount of instances the AccelerationStructure will hold
    */
    AccelerationStructure(ContextHandle context, uint32_t capacity);
    /**
     * @brief Creates a new acceleration structure
     * 
     * @param context Context on which to create the AccelerationStructure
     * @param instance GeometryInstance used to create the AccelerationStructure
     * @param frozen If false, allows the acceleration structure to be modified
    */
    AccelerationStructure(ContextHandle context, const GeometryInstance& instance, bool frozen = true);
    /**
     * @brief Creates a new acceleration structure
     * 
     * @param context Context on which to create the AccelerationStructure
     * @param instances List of GeometryInstance used to create the
     *                  AccelerationStructure
    */
    AccelerationStructure(ContextHandle context, std::span<const GeometryInstance> instances, bool frozen = true);
    ~AccelerationStructure() override;

protected:
    void onDestroy() override;

private:
    AccelerationStructure(ContextHandle context, const GeometryInstance* pInstances, size_t instanceCount, bool frozen);

    struct BuildResources;
    std::unique_ptr<BuildResources> buildResources;

    struct Parameter;
    std::unique_ptr<Parameter> param;

    friend class BuildAccelerationStructureCommand;
};

/**
 * @brief Command issuing the rebuild of the corresponding acceleration structure.
 */
class HEPHAISTOS_API BuildAccelerationStructureCommand : public Command {
public:
    /**
     * @brief Acceleration structure to be updated
     */
    std::reference_wrapper<const AccelerationStructure> accelerationStructure;
    /**
     * @brief If true, omits barriers protecting read and write of associated buffers
     */
    bool unsafe;

    virtual void record(vulkan::Command& cmd) const override;

    BuildAccelerationStructureCommand(const BuildAccelerationStructureCommand& other);
    BuildAccelerationStructureCommand& operator=(const BuildAccelerationStructureCommand& other);

    BuildAccelerationStructureCommand(BuildAccelerationStructureCommand&& other) noexcept;
    BuildAccelerationStructureCommand& operator=(BuildAccelerationStructureCommand&& other) noexcept;

    /**
     * @brief Creates a new BuildAccelerationStructureCommand
     * 
     * @param accelerationStructure AccelerationStructure to be updated
     * @param unsafe Whether to omit memory protection barriers
     */
    BuildAccelerationStructureCommand(
        const AccelerationStructure& accelerationStructure,
        bool unsafe = false);
    ~BuildAccelerationStructureCommand() override;
};
/**
 * @brief Creates a BuildAccelerationStructureCommand
 * 
 * @param accelerationStructure Acceleration structure to be rebuild
 * @param unsafe Whether to skip memory barriers
 */
[[nodiscard]] inline BuildAccelerationStructureCommand buildAccelerationStructure(
    const AccelerationStructure& accelerationStructure,
    bool unsafe = false
) {
    return BuildAccelerationStructureCommand(accelerationStructure, unsafe);
}

}
