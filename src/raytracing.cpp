#include "hephaistos/raytracing.hpp"

#include <algorithm>
#include <array>
#include <cstring>

#include "volk.h"

#include "vk/types.hpp"
#include "vk/result.hpp"
#include "vk/util.hpp"

namespace hephaistos {

/********************************** EXTENSION *********************************/

namespace {

constexpr auto ExtensionName = "Raytracing";

//! MUST BE SORTED FOR std::includes !//
constexpr auto DeviceExtensions = std::to_array({
    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
    VK_KHR_RAY_QUERY_EXTENSION_NAME,
    VK_KHR_RAY_TRACING_POSITION_FETCH_EXTENSION_NAME
});

}

bool isRaytracingSupported(const DeviceHandle& device) {
    //We have to check two things:
    // - Extensions supported
    // - Features suppported

    //nullcheck
    if (!device)
        return false;

    //Check extension support
    if (!std::includes(
        device->supportedExtensions.begin(),
        device->supportedExtensions.end(),
        DeviceExtensions.begin(),
        DeviceExtensions.end()))
    {
        return false;
    }

    //Query features
    VkPhysicalDeviceRayTracingPositionFetchFeaturesKHR posFetchFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_POSITION_FETCH_FEATURES_KHR,
    };
    VkPhysicalDeviceRayQueryFeaturesKHR queryFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR,
        .pNext = &posFetchFeatures
    };
    VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
        .pNext = &queryFeatures
    };
    VkPhysicalDeviceFeatures2 features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &accelerationStructureFeatures
    };
    vkGetPhysicalDeviceFeatures2(device->device, &features);

    //check features
    if (!posFetchFeatures.rayTracingPositionFetch)
        return false;
    if (!queryFeatures.rayQuery)
        return false;
    if (!accelerationStructureFeatures.accelerationStructure)
        return false;

    //Everything checked
    return true;
}
bool isRaytracingEnabled(const ContextHandle& context) {
    //to shorten things
    auto& ext = context->extensions;
    return std::find_if(ext.begin(), ext.end(),
        [](const ExtensionHandle& h) -> bool {
            return h->getExtensionName() == ExtensionName;
        }) != ext.end();
}

class RaytracingExtension : public Extension {
public:
    bool isDeviceSupported(const DeviceHandle& device) const override {
        return isRaytracingSupported(device);
    }
    std::string_view getExtensionName() const override {
        return ExtensionName;
    }
    std::span<const char* const> getDeviceExtensions() const override {
        return DeviceExtensions;
    }
    void* chain(void* pNext) override {
        queryFeatures.pNext = pNext;
        return static_cast<void*>(&accelerationStructureFeatures);
    }

    RaytracingExtension() = default;
    virtual ~RaytracingExtension() = default;

private:
    VkPhysicalDeviceRayQueryFeaturesKHR queryFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR,
        .rayQuery = VK_TRUE
    };
    VkPhysicalDeviceRayTracingPositionFetchFeaturesKHR positionFetchFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_POSITION_FETCH_FEATURES_KHR,
        .pNext = &queryFeatures,
        .rayTracingPositionFetch = VK_TRUE
    };
    VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
        .pNext = &positionFetchFeatures,
        .accelerationStructure = VK_TRUE
    };
};
ExtensionHandle createRaytracingExtension() {
    return std::make_unique<RaytracingExtension>();
}

/********************************* GEOMETRIES *********************************/

struct GeometryStore::Imp {
    std::vector<Geometry> geometries{};
    std::vector<VkAccelerationStructureKHR> blas{};

    BufferHandle dataBuffer = vulkan::createEmptyBuffer();
    BufferHandle blasBuffer = vulkan::createEmptyBuffer();
};

const std::vector<Geometry>& GeometryStore::geometries() const noexcept {
    return pImp->geometries;
}

const Geometry& GeometryStore::operator[](size_t idx) const {
    return pImp->geometries[idx];
}

size_t GeometryStore::size() const noexcept {
    return pImp->geometries.size();
}

GeometryInstance GeometryStore::createInstance(
    size_t idx,
    const TransformMatrix& transform,
    uint32_t customIndex,
    uint8_t mask) const
{
    return {
        (*this)[idx].blas_address,
        transform,
        customIndex,
        mask
    };
}

void GeometryStore::onDestroy() {
    if (pImp) {
        auto& context = *getContext();
        for (auto acc : pImp->blas) {
            context.fnTable.vkDestroyAccelerationStructureKHR(
                context.device, acc, nullptr);
        }
        pImp.reset();
    }
}

GeometryStore::GeometryStore(GeometryStore&&) noexcept = default;
GeometryStore& GeometryStore::operator=(GeometryStore&&) = default;

GeometryStore::GeometryStore(
    ContextHandle context,
    const Mesh& mesh,
    bool keepGeometryData)
    : GeometryStore(std::move(context), { &mesh, 1 }, keepGeometryData)
{}
GeometryStore::GeometryStore(
    ContextHandle _context,
    std::span<const Mesh> meshes,
    bool keepGeometryData)
    : Resource(std::move(_context))
    , pImp(std::make_unique<Imp>())
{
    auto& context = getContext();
    auto nMeshes = meshes.size();
    //Calculate how much memory we'll need for the geometry data
    uint64_t total_vertices_size = 0;
    uint64_t total_indices_size = 0;
    for (auto& m : meshes) {
        total_vertices_size += m.vertices.size_bytes();
        total_indices_size += m.indices.size_bytes();
    }
    auto hasIndices = total_indices_size > 0;
    //Not sure if needed, but ensure 4 byte alignment for indices
    if (hasIndices && total_vertices_size % 4) {
        total_vertices_size += 4 - (total_vertices_size % 4);
    }
    auto data_size = total_vertices_size + total_indices_size;

    //make data visible to gpu
    auto dataBuffer = vulkan::createBuffer(context,
        data_size,
        keepGeometryData ?
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT : 
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
        VMA_ALLOCATION_CREATE_MAPPED_BIT);
    {
        //copy data to buffer
        auto p = static_cast<std::byte*>(dataBuffer->allocInfo.pMappedData);
        for (auto& m : meshes) {
            std::memcpy(p, m.vertices.data(), m.vertices.size_bytes());
            p += m.vertices.size_bytes();
        }
        if (hasIndices) {
            //respect padding
            p = static_cast<std::byte*>(dataBuffer->allocInfo.pMappedData) + total_vertices_size;
            for (auto& m : meshes) {
                if (m.indices.empty())
                    continue;
                std::memcpy(p, m.indices.data(), m.indices.size_bytes());
                p += m.indices.size_bytes();
            }
        }
    }
    if (keepGeometryData) {
        //use buffer as staging and upload to gpu local
        auto gpuBuffer = vulkan::createBuffer(context,
            data_size,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            0);
        //upload data
        vulkan::oneTimeSubmit(*context, [&](VkCommandBuffer cmd) {
            VkBufferCopy copyRegion{
                .size = data_size
            };
            context->fnTable.vkCmdCopyBuffer(
                cmd, dataBuffer->buffer, gpuBuffer->buffer, 1, &copyRegion);
        });
        //replace dataBuffer
        dataBuffer = std::move(gpuBuffer);
    }

    //query buffer addresses
    VkDeviceAddress vertexAddress, indexAddress = 0;
    {
        VkBufferDeviceAddressInfo addressInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
            .buffer = dataBuffer->buffer
        };
        vertexAddress = context->fnTable.vkGetBufferDeviceAddress(
            context->device, &addressInfo);
        if (hasIndices) {
            indexAddress = vertexAddress + total_vertices_size;
        }
    }

    //query scratch buffer alignment
    uint32_t scratchAlignment;
    {
        VkPhysicalDeviceAccelerationStructurePropertiesKHR accProps{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR
        };
        VkPhysicalDeviceProperties2 props{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            .pNext = &accProps
        };
        vkGetPhysicalDeviceProperties2(context->physicalDevice, &props);
        scratchAlignment = accProps.minAccelerationStructureScratchOffsetAlignment;
    }

    //fill blas info
    VkDeviceSize blasTotalSize = 0;
    VkDeviceSize totalScratchSize = 0;
    auto pVertex = vertexAddress;
    auto pIndex = indexAddress;
    std::vector<VkAccelerationStructureGeometryKHR> geometries(nMeshes);
    std::vector<VkAccelerationStructureBuildGeometryInfoKHR> buildInfo(nMeshes);
    std::vector<VkAccelerationStructureBuildSizesInfoKHR> sizes(nMeshes);
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> ranges(nMeshes);
    for (auto i = 0u; i < nMeshes; ++i) {
        auto& geometry = meshes[i];
        uint32_t vertex_count = geometry.vertices.size_bytes() / geometry.vertexStride;
        auto hasIdx = !geometry.indices.empty();
        //fill geometry info
        VkAccelerationStructureGeometryTrianglesDataKHR triangles{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
            .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
            .vertexData = { .deviceAddress = pVertex },
            .vertexStride = geometry.vertexStride,
            .maxVertex = vertex_count - 1,
            .indexType = hasIdx ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_NONE_KHR,
            .indexData = { .deviceAddress = hasIdx ? pIndex : 0 }
        };
        geometries[i] = {
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
            .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
            .geometry = { .triangles = triangles },
            .flags = VK_GEOMETRY_OPAQUE_BIT_KHR
        };
        //update addresses
        pVertex += meshes[i].vertices.size_bytes();
        pIndex += meshes[i].indices.size_bytes();

        //fill size query
        buildInfo[i] = {
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
            .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
            .flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                     VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR |
                     VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_DATA_ACCESS_KHR,
            .geometryCount = 1,
            .pGeometries = &geometries[i]
        };
        sizes[i] = {
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR
        };
        //calculate number of triangles
        uint32_t triangleCount = (hasIdx ? geometry.indices.size() : vertex_count) / 3;
        ranges[i].primitiveCount = triangleCount;
        //fetch build size
        context->fnTable.vkGetAccelerationStructureBuildSizesKHR(
            context->device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
            &buildInfo[i], &triangleCount, &sizes[i]);

        //update sizes
        blasTotalSize += sizes[i].accelerationStructureSize;
        if (blasTotalSize % 256) //force allignment
            blasTotalSize += 256 - (blasTotalSize % 256);
        totalScratchSize += sizes[i].buildScratchSize;
        if (totalScratchSize % scratchAlignment)
            totalScratchSize += scratchAlignment - (totalScratchSize % scratchAlignment);
    }

    //Create scratch buffer
    auto scratchBuffer = vulkan::createBufferAligned(context,
        totalScratchSize, scratchAlignment,
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        0);
    auto scratchAddress = vulkan::getBufferDeviceAddress(scratchBuffer);
    //allocate buffer for acceleration structures
    auto blasBuffer = vulkan::createBuffer(context, blasTotalSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

    //create query pool for compacted blas sizes
    VkQueryPool queryPool;
    {
        VkQueryPoolCreateInfo poolInfo{
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
            .queryCount = static_cast<uint32_t>(nMeshes)
        };
        vulkan::checkResult(context->fnTable.vkCreateQueryPool(
            context->device, &poolInfo, nullptr, &queryPool));
        context->fnTable.vkResetQueryPool(context->device, queryPool, 0, nMeshes);
    }

    //create blas
    VkDeviceSize offset = 0;
    std::vector<VkAccelerationStructureKHR> accStructures(nMeshes);
    for (auto i = 0u; i < nMeshes; ++i) {
        //create acceleration structure
        VkAccelerationStructureCreateInfoKHR accInfo{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
            .buffer = blasBuffer->buffer,
            .offset = offset,
            .size = sizes[i].accelerationStructureSize,
            .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR
        };
        offset += accInfo.size;
        if (accInfo.size % 256)
            offset += 256 - (accInfo.size % 256);
        vulkan::checkResult(context->fnTable.vkCreateAccelerationStructureKHR(
            context->device, &accInfo, nullptr, &accStructures[i]));
        //register
        buildInfo[i].dstAccelerationStructure = accStructures[i];
        buildInfo[i].scratchData.deviceAddress = scratchAddress;
        //update address
        scratchAddress += sizes[i].buildScratchSize;
        if (scratchAddress % scratchAlignment)
            scratchAddress += scratchAlignment - (scratchAddress % scratchAlignment);
    }
    //build blas
    vulkan::oneTimeSubmit(*context, [&](VkCommandBuffer cmd) {
        //build 2d array (second dimension is 1...)
        std::vector<const VkAccelerationStructureBuildRangeInfoKHR*> pRanges(ranges.size());
        std::transform(ranges.begin(), ranges.end(), pRanges.begin(),
            [](const VkAccelerationStructureBuildRangeInfoKHR& i) { return &i; });
        //issue build
        context->fnTable.vkCmdBuildAccelerationStructuresKHR(cmd,
            static_cast<uint32_t>(buildInfo.size()),
            buildInfo.data(),
            pRanges.data());
        //memory barrier to ensure queries are valid
        VkMemoryBarrier barrier{
            .sType           = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
            .dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR
        };
        context->fnTable.vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            0, 1, &barrier, 0, nullptr, 0, nullptr);
        //query
        context->fnTable.vkCmdWriteAccelerationStructuresPropertiesKHR(cmd,
            nMeshes, accStructures.data(),
            VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
            queryPool, 0);
    });

    //query compacted sizes
    std::vector<VkDeviceSize> compactSizes(nMeshes);
    context->fnTable.vkGetQueryPoolResults(
        context->device,
        queryPool,
        0, nMeshes,
        nMeshes * sizeof(VkDeviceSize),
        compactSizes.data(),
        sizeof(VkDeviceSize),
        VK_QUERY_RESULT_WAIT_BIT);
    //destroy query pool
    context->fnTable.vkDestroyQueryPool(context->device, queryPool, nullptr);
    
    //update blas buffer size
    blasTotalSize = 0;
    for (auto size : compactSizes) {
        blasTotalSize += size;
        if (size % 256)
            blasTotalSize += 256 - (size % 256);
    }
    //create new compact blas buffer
    auto compactBlasBuffer = vulkan::createBuffer(context, blasTotalSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    //create compacted blas
    offset = 0;
    std::vector<VkAccelerationStructureKHR> compactAccStructures(nMeshes);
    for (auto i = 0u; i < nMeshes; ++i) {
        //create acceleration structure
        VkAccelerationStructureCreateInfoKHR accInfo{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
            .buffer = compactBlasBuffer->buffer,
            .offset = offset,
            .size = compactSizes[i],
            .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR
        };
        offset += accInfo.size;
        if (accInfo.size % 256)
            offset += 256 - (accInfo.size % 256);
        vulkan::checkResult(context->fnTable.vkCreateAccelerationStructureKHR(
            context->device, &accInfo, nullptr, &compactAccStructures[i]));
    }
    //copy blas
    vulkan::oneTimeSubmit(*context, [&](VkCommandBuffer cmd) {
        VkCopyAccelerationStructureInfoKHR copyInfo{
            .sType = VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR,
            .mode  = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR
        };
        for (auto i = 0u; i < nMeshes; ++i) {
            copyInfo.src = accStructures[i];
            copyInfo.dst = compactAccStructures[i];
            context->fnTable.vkCmdCopyAccelerationStructureKHR(cmd, &copyInfo);
        }
    });
    //destroy inital blas
    for (auto i = 0u; i < nMeshes; ++i) {
        context->fnTable.vkDestroyAccelerationStructureKHR(
            context->device, accStructures[i], nullptr);
    }

    //Done -> build result
    pVertex = vertexAddress;
    pIndex = indexAddress;
    std::vector<Geometry> blasResult(nMeshes);
    for (auto i = 0u; i < nMeshes; ++i) {
        if (keepGeometryData) {
            blasResult[i].vertices_address = pVertex;
            blasResult[i].indices_address = meshes[i].indices.empty() ? 0 : pIndex;
            pVertex += meshes[i].vertices.size_bytes();
            pIndex += meshes[i].indices.size_bytes();
        }
        //fetch device address
        VkAccelerationStructureDeviceAddressInfoKHR addressInfo{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
            .accelerationStructure = compactAccStructures[i]
        };
        blasResult[i].blas_address = context->fnTable.vkGetAccelerationStructureDeviceAddressKHR(
            context->device, &addressInfo);
    }
    //save results
    pImp->blas = std::move(compactAccStructures);
    pImp->blasBuffer = std::move(compactBlasBuffer);
    pImp->geometries = std::move(blasResult);
    if (keepGeometryData) {
        pImp->dataBuffer = std::move(dataBuffer);
    }
}

GeometryStore::~GeometryStore() {
    onDestroy();
}

/*************************** ACCELERATION STRUCTURE ***************************/

namespace {

constexpr VkBufferUsageFlags TLAS_BUFFER_USAGE_FLAGS =
    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

constexpr VkBufferUsageFlags INSTANCE_BUFFER_USAGE_FLAGS =
    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
    // VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
    // VK_BUFFER_USAGE_TRANSFER_DST_BIT |
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

constexpr VmaAllocationCreateFlags INSTANCE_BUFFER_ALLOCATION_FLAGS =
    VMA_ALLOCATION_CREATE_MAPPED_BIT |
    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

constexpr VkBufferUsageFlags SCRATCH_BUFFER_USAGE_FLAGS =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

const GeometryInstance defaultGeometryInstance = {};

}

struct AccelerationStructure::BuildResources {
    BufferHandle instanceBuffer = vulkan::createEmptyBuffer();
    BufferHandle scratchBuffer = vulkan::createEmptyBuffer();

    VkDeviceAddress instanceBufferAddress = 0;
    std::span<VkAccelerationStructureInstanceKHR> instances = {};

    VkAccelerationStructureGeometryKHR tlasGeometry = {};
    VkAccelerationStructureBuildGeometryInfoKHR tlasGeometryInfo = {};
    VkAccelerationStructureBuildRangeInfoKHR tlasRangeInfo = {};
    const VkAccelerationStructureBuildRangeInfoKHR* pRangeInfo = &tlasRangeInfo;

    VkDeviceSize init(
        const ContextHandle& context,
        const GeometryInstance* pInstances,
        uint32_t instanceCount
    );

    void build(const ContextHandle& context);
};

struct AccelerationStructure::Parameter {
    VkAccelerationStructureKHR tlas = 0;
    VkWriteDescriptorSetAccelerationStructureKHR descriptorInfo{};

    uint32_t capacity = 0;
    uint32_t instanceCount = 0;

    BufferHandle tlasBuffer = vulkan::createEmptyBuffer();

    void init(
        const ContextHandle& context,
        BuildResources& buildResources,
        VkDeviceSize tlasSize
    );
};

VkDeviceSize AccelerationStructure::BuildResources::init(
    const ContextHandle& context,
    const GeometryInstance* pInstances,
    uint32_t instanceCount
) {
    //create instance buffer
    instanceBuffer = vulkan::createBufferAligned(
        context,
        instanceCount * sizeof(VkAccelerationStructureInstanceKHR),
        16,
        INSTANCE_BUFFER_USAGE_FLAGS,
        INSTANCE_BUFFER_ALLOCATION_FLAGS
    );
    instanceBufferAddress = vulkan::getBufferDeviceAddress(instanceBuffer);
    instances = {
        static_cast<VkAccelerationStructureInstanceKHR*>(instanceBuffer->allocInfo.pMappedData),
        instanceCount
    };
    //fill instance buffer
    auto pInstance = &defaultGeometryInstance;
    auto pOut = instances.data();
    for (auto i = 0u; i < instanceCount; ++i, ++pOut) {
        //point to actual data if we have any
        if (pInstances)
            pInstance = &pInstances[i];

        //copy transform matrix
        memcpy(pOut, &pInstance->transform, sizeof(VkTransformMatrixKHR));
        //set rest
        pOut->instanceCustomIndex = pInstance->customIndex;
        pOut->mask = pInstance->mask;
        pOut->instanceShaderBindingTableRecordOffset = 0;
        pOut->flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        pOut->accelerationStructureReference = pInstance->blas_address;
    }

    //create tlas geometry
    tlasGeometry = {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
        .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
        .flags = VK_GEOMETRY_OPAQUE_BIT_KHR
    };
    tlasGeometry.geometry.instances = {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
        .arrayOfPointers = VK_FALSE,
        .data = {.deviceAddress = instanceBufferAddress }
    };
    tlasRangeInfo = { .primitiveCount = instanceCount };

    //get size info
    tlasGeometryInfo = {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
        .type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
        .flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                 VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_DATA_ACCESS_KHR,
        .geometryCount = 1,
        .pGeometries = &tlasGeometry
    };
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR
    };
    context->fnTable.vkGetAccelerationStructureBuildSizesKHR(
        context->device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &tlasGeometryInfo, &instanceCount, &sizeInfo
    );
    //query scratch buffer alignment
    uint32_t scratchAlignment;
    {
        VkPhysicalDeviceAccelerationStructurePropertiesKHR accProps{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR
        };
        VkPhysicalDeviceProperties2 props{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            .pNext = &accProps
        };
        vkGetPhysicalDeviceProperties2(context->physicalDevice, &props);
        scratchAlignment = accProps.minAccelerationStructureScratchOffsetAlignment;
    }
    //create scratch buffer
    scratchBuffer = vulkan::createBufferAligned(
        context,
        sizeInfo.buildScratchSize,
        scratchAlignment,
        SCRATCH_BUFFER_USAGE_FLAGS,
        0
    );
    tlasGeometryInfo.scratchData.deviceAddress = vulkan::getBufferDeviceAddress(scratchBuffer);
    //return tlas size so we do not need to store it
    return sizeInfo.accelerationStructureSize;
}

void AccelerationStructure::BuildResources::build(const ContextHandle& context) {
    vulkan::oneTimeSubmit(*context, [this, &context](VkCommandBuffer cmd) {
        context->fnTable.vkCmdBuildAccelerationStructuresKHR(cmd, 1, &tlasGeometryInfo, &pRangeInfo);
    });
}

void AccelerationStructure::Parameter::init(
    const ContextHandle& context,
    AccelerationStructure::BuildResources& buildResources,
    VkDeviceSize tlasSize
) {
    capacity = buildResources.instances.size();
    instanceCount = buildResources.instances.size();
    //create acceleration structure
    tlasBuffer = vulkan::createBuffer(
        context,
        tlasSize,
        TLAS_BUFFER_USAGE_FLAGS,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
    );
    VkAccelerationStructureCreateInfoKHR tlasInfo{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
        .buffer = tlasBuffer->buffer,
        .size = tlasSize,
        .type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR
    };
    vulkan::checkResult(context->fnTable.vkCreateAccelerationStructureKHR(
        context->device, &tlasInfo, nullptr, &tlas
    ));
    buildResources.tlasGeometryInfo.dstAccelerationStructure = tlas;

    //building tlas is handled by BuildResources::build()

    //build descriptor info for binding tlas
    descriptorInfo = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
        .accelerationStructureCount = 1,
        .pAccelerationStructures = &tlas
    };
}

void AccelerationStructure::bindParameter(VkWriteDescriptorSet& binding) const {
    binding.pNext = &param->descriptorInfo;
    binding.pBufferInfo = nullptr;
    binding.pImageInfo = nullptr;
    binding.pTexelBufferView = nullptr;
}

uint32_t AccelerationStructure::capacity() const noexcept {
    return param->capacity;
}
uint32_t AccelerationStructure::size() const noexcept {
    return param->instanceCount;
}

uint64_t AccelerationStructure::instanceBufferAddress() const {
    if (frozen())
        throw std::runtime_error("Cannot fetch instance buffer address of frozen acceleration structure!");

    return buildResources->instanceBufferAddress;
}

void AccelerationStructure::update(std::span<const GeometryInstance> instances) {
    if (frozen())
        throw std::runtime_error("Cannot update frozen acceleration structure!");
    if (instances.size() > buildResources->instances.size())
        throw std::runtime_error("Too many instances to fit in acceleration structure!");
    
    //copy instances to buffer
    auto pOut = buildResources->instances.data();
    auto n = buildResources->instances.size();
    for (auto i = 0; i < n; ++i, ++pOut) {
        //either fetch next instance or use default to make next one inactive
        const auto& instance = i < instances.size() ? instances[i] : defaultGeometryInstance;
        //copy instance
        memcpy(pOut, &instance.transform, sizeof(VkTransformMatrixKHR));
        pOut->instanceCustomIndex = instance.customIndex;
        pOut->mask = instance.mask;
        pOut->accelerationStructureReference = instance.blas_address;
        //other fields should still be valid
    }
    //rebuild tlas
    buildResources->build(getContext());
    param->instanceCount = instances.size();
}

bool AccelerationStructure::frozen() const noexcept {
    return !buildResources;
}
void AccelerationStructure::freeze() {
    //free build resources, we no longer need it
    buildResources.reset();
}

void AccelerationStructure::onDestroy() {
    if (param) {
        auto& context = *getContext();
        context.fnTable.vkDestroyAccelerationStructureKHR(
            context.device, param->tlas, nullptr);
        param.reset();
    }
}

AccelerationStructure::AccelerationStructure(AccelerationStructure&&) noexcept = default;
AccelerationStructure& AccelerationStructure::operator=(AccelerationStructure&&) = default;

AccelerationStructure::AccelerationStructure(ContextHandle context, uint32_t capacity)
    : AccelerationStructure(std::move(context), nullptr, capacity, false)
{
    param->instanceCount = 0;
}
AccelerationStructure::AccelerationStructure(ContextHandle context, const GeometryInstance& instance, bool frozen)
    : AccelerationStructure(std::move(context), &instance, 1, frozen)
{}
AccelerationStructure::AccelerationStructure(ContextHandle context, std::span<const GeometryInstance> instances, bool frozen)
    : AccelerationStructure(std::move(context), instances.data(), instances.size(), frozen)
{}
AccelerationStructure::AccelerationStructure(
    ContextHandle context,
    const GeometryInstance* pInstances,
    size_t instanceCount,
    bool frozen
)
    : Resource(std::move(context))
    , buildResources(std::make_unique<BuildResources>())
    , param(std::make_unique<Parameter>())
{
    auto tlasSize = buildResources->init(getContext(), pInstances, instanceCount);
    param->init(getContext(), *buildResources, tlasSize);
    buildResources->build(getContext());

    if (frozen) freeze();
}

AccelerationStructure::~AccelerationStructure() {
    onDestroy();
}

void BuildAccelerationStructureCommand::record(vulkan::Command& cmd) const {
    //to shorten things
    auto& context = accelerationStructure.get().getContext();
    auto& buildResources = accelerationStructure.get().buildResources;
    auto instanceBuffer = buildResources->instanceBuffer->buffer;
    auto scratchBuffer = buildResources->scratchBuffer->buffer;

    //we're acting on the acceleration structure build stage
    cmd.stage |= VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;

    //ensure previous memory operation have finished
    if (!unsafe) {
        //acceleration structure barrier
        //TODO: Not sure if a buffer barrier on the underlying buffer would also work?
        VkMemoryBarrier memoryBarrier{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
            .dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR
        };
        auto bufferBarriers = std::to_array<VkBufferMemoryBarrier>({
            //scratch buffer
            {
                .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                .buffer = scratchBuffer,
                .size = VK_WHOLE_SIZE
            },
            //instance buffer
            {
                .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
                .buffer = instanceBuffer,
                .size = VK_WHOLE_SIZE
            }
        });
        context->fnTable.vkCmdPipelineBarrier(cmd.buffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            VK_DEPENDENCY_BY_REGION_BIT,
            1, &memoryBarrier,
            static_cast<uint32_t>(bufferBarriers.size()), bufferBarriers.data(),
            0, nullptr);
    }

    //build acceleration structure
    context->fnTable.vkCmdBuildAccelerationStructuresKHR(cmd.buffer,
        1, &buildResources->tlasGeometryInfo, &buildResources->pRangeInfo);

    //block subsequent operations that may use the acceleration structure until building it finishes
    if (!unsafe) {
        //acceleration structure barrier
        //TODO: Not sure if a buffer barrier on the underlying buffer would also work?
        VkMemoryBarrier memoryBarrier{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
            .dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR
        };
        //ensure we aren't changing instance buffer while building acceleration structure
        VkBufferMemoryBarrier bufferBarrier{
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
            .dstAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
            .buffer = instanceBuffer,
            .size = VK_WHOLE_SIZE
        };
        context->fnTable.vkCmdPipelineBarrier(cmd.buffer,
            VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_DEPENDENCY_BY_REGION_BIT,
            1, &memoryBarrier,
            1, &bufferBarrier,
            0, nullptr);
    }
}

BuildAccelerationStructureCommand::BuildAccelerationStructureCommand(
    const BuildAccelerationStructureCommand& other
) = default;
BuildAccelerationStructureCommand& BuildAccelerationStructureCommand::operator=(
    const BuildAccelerationStructureCommand& other
) = default;

BuildAccelerationStructureCommand::BuildAccelerationStructureCommand(
    BuildAccelerationStructureCommand&& other
) noexcept = default;
BuildAccelerationStructureCommand& BuildAccelerationStructureCommand::operator=(
    BuildAccelerationStructureCommand&& other
) noexcept = default;

BuildAccelerationStructureCommand::BuildAccelerationStructureCommand(
    const AccelerationStructure& accelerationStructure,
    bool unsafe
) : accelerationStructure(std::cref(accelerationStructure)), unsafe(unsafe)
{}
BuildAccelerationStructureCommand::~BuildAccelerationStructureCommand() = default;

}
