#include "hephaistos/raytracing.hpp"

#include <algorithm>
#include <array>
#include <cstring>

#include "volk.h"

#include "vk/types.hpp"
#include "vk/result.hpp"
#include "vk/util.hpp"

namespace hephaistos {

/*********************************** EXTENSION ********************************/

namespace {

constexpr auto ExtensionName = "Raytracing";

//! MUST BE SORTED FOR std::includes !//
constexpr auto DeviceExtensions = std::to_array({
	VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
	VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
	VK_KHR_RAY_QUERY_EXTENSION_NAME
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
	VkPhysicalDeviceRayQueryFeaturesKHR queryFeatures{
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR,
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
	std::string_view name{ ExtensionName };
	return std::find(ext.begin(), ext.end(), name) != ext.end();
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
	VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeatures{
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
		.pNext = &queryFeatures,
		.accelerationStructure = VK_TRUE
	};
};
ExtensionHandle createRaytracingExtension() {
	return std::make_unique<RaytracingExtension>();
}

/*************************** ACCELERATION STRUCTURE ***************************/

namespace vulkan {

struct AccelerationStructure {
	VkAccelerationStructureKHR handle;
	uint64_t deviceAddress = 0;
	VkBuffer buffer;
	VmaAllocation allocation;
};

void createAccelerationStructure(
	const Context& context,
	AccelerationStructure& acc,
	VkAccelerationStructureTypeKHR type,
	VkAccelerationStructureBuildSizesInfoKHR& info)
{
	//create buffer
	VkBufferCreateInfo bufferInfo{
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = info.accelerationStructureSize,
		.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
	};
	VmaAllocationCreateInfo allocInfo{
		.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
	};
	checkResult(vmaCreateBuffer(
		context.allocator,
		&bufferInfo, &allocInfo,
		&acc.buffer, &acc.allocation,
		nullptr));

	//Create acceleration structure
	VkAccelerationStructureCreateInfoKHR accInfo{
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
		.buffer = acc.buffer,
		.size = info.accelerationStructureSize,
		.type = type
	};
	checkResult(context.fnTable.vkCreateAccelerationStructureKHR(
		context.device, &accInfo, nullptr, &acc.handle));

	//fetch device address
	VkAccelerationStructureDeviceAddressInfoKHR addressInfo{
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
		.accelerationStructure = acc.handle
	};
	acc.deviceAddress = context.fnTable.vkGetAccelerationStructureDeviceAddressKHR(
		context.device, &addressInfo);
}

void destroyAccelerationStructure(const Context& context, AccelerationStructure& acc) {
	context.fnTable.vkDestroyAccelerationStructureKHR(context.device, acc.handle, nullptr);
	vmaDestroyBuffer(context.allocator, acc.buffer, acc.allocation);
}

}

struct AccelerationStructure::Parameter {
	std::vector<vulkan::AccelerationStructure> blas;
	vulkan::AccelerationStructure tlas;

	VkWriteDescriptorSetAccelerationStructureKHR descriptorInfo;

	vulkan::Context& context;

	Parameter(vulkan::Context& context, std::span<const GeometryInstance> geometries);
	~Parameter();

private:
	vulkan::AccelerationStructure createBlas(const Geometry& geometry);
	void createTlas(std::span<const GeometryInstance> instances, std::span<size_t> blasIdx);
};

vulkan::AccelerationStructure AccelerationStructure::Parameter::createBlas(const Geometry& inGeo) {
	vulkan::AccelerationStructure result{};

	//create device local buffer for vertex and instance
	VkBuffer inputBuffer;
	VkDeviceAddress inputAddress;
	VmaAllocation inputBufferAllocation;
	VmaAllocationInfo inputBufferInfo;
	{
		//create buffer
		VkBufferCreateInfo bufferInfo{
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = inGeo.vertices.size_bytes() + inGeo.indices.size_bytes(),
			.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
				VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
		};
		VmaAllocationCreateInfo allocInfo{
			.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
				VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
			.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
		};
		vulkan::checkResult(vmaCreateBuffer(
			context.allocator,
			&bufferInfo, &allocInfo,
			&inputBuffer, &inputBufferAllocation, &inputBufferInfo));

		//get address
		VkBufferDeviceAddressInfo addressInfo{
			.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
			.buffer = inputBuffer
		};
		inputAddress = context.fnTable.vkGetBufferDeviceAddress(
			context.device, &addressInfo);

		//copy data
		auto pointer = static_cast<std::byte*>(inputBufferInfo.pMappedData);
		std::memcpy(pointer, inGeo.vertices.data(), inGeo.vertices.size_bytes());
		pointer += inGeo.vertices.size_bytes();
		std::memcpy(pointer, inGeo.indices.data(), inGeo.indices.size_bytes());
	}

	//fill in build info / data
	VkAccelerationStructureGeometryKHR geometry{};
	VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
	VkAccelerationStructureBuildSizesInfoKHR sizes{};
	VkAccelerationStructureBuildRangeInfoKHR range{};
	{
		//geometry
		geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
		geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
		geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
		//geometry data
		geometry.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
		geometry.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
		geometry.geometry.triangles.vertexData.deviceAddress = inputAddress;
		geometry.geometry.triangles.vertexStride = inGeo.vertexStride;
		geometry.geometry.triangles.maxVertex = inGeo.vertexCount;
		if (!inGeo.indices.empty()) {
			geometry.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
			geometry.geometry.triangles.indexData.deviceAddress = inputAddress + inGeo.vertices.size_bytes();
		}
		//build info
		buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
		buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
		buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
		//VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
		buildInfo.geometryCount = 1;
		buildInfo.pGeometries = &geometry;
		//sizes
		sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
	}

	//calculate number of triangles
	uint32_t triangleCount = inGeo.indices.empty() ? inGeo.vertexCount : inGeo.indices.size();
	triangleCount /= 3;
	range.primitiveCount = triangleCount;

	//fetch build sizes
	context.fnTable.vkGetAccelerationStructureBuildSizesKHR(
		context.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
		&buildInfo, &triangleCount, &sizes);

	//create accelerations structure
	vulkan::createAccelerationStructure(context, result,
		VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, sizes);
	//register handle
	buildInfo.dstAccelerationStructure = result.handle;

	//Create scratch buffer
	VkBuffer scratchBuffer;
	VmaAllocation scratchAllocation;
	VkDeviceAddress scratchAddress;
	{
		//allocate buffer
		VkBufferCreateInfo bufferInfo{
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = sizes.buildScratchSize,
			.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
		};
		VmaAllocationCreateInfo allocInfo{
			.usage = VMA_MEMORY_USAGE_AUTO
		};
		vulkan::checkResult(vmaCreateBuffer(context.allocator,
			&bufferInfo, &allocInfo,
			&scratchBuffer, &scratchAllocation, nullptr));
		//get device address
		VkBufferDeviceAddressInfo addressInfo{
			.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
			.buffer = scratchBuffer
		};
		scratchAddress = context.fnTable.vkGetBufferDeviceAddress(context.device, &addressInfo);
	}
	//register scratch buffer
	buildInfo.scratchData.deviceAddress = scratchAddress;

	//record build command
	auto& c = context;
	vulkan::oneTimeSubmit(context, [&c, &range, &buildInfo](VkCommandBuffer cmd) {
		//issue build
		auto rangeInfo = &range;
		c.fnTable.vkCmdBuildAccelerationStructuresKHR(cmd,
			1, &buildInfo, &rangeInfo);
	});

	//destroy buffers
	vmaDestroyBuffer(context.allocator, scratchBuffer, scratchAllocation);
	vmaDestroyBuffer(context.allocator, inputBuffer, inputBufferAllocation);

	//done
	return result;
}

void AccelerationStructure::Parameter::createTlas(
	std::span<const GeometryInstance> instances,
	std::span<size_t> blasIdx)
{
	//create instance buffer
	VkBuffer instanceBuffer;
	VmaAllocation instanceBufferAllocation;
	VmaAllocationInfo instanceBufferInfo;
	VkDeviceAddress instanceBufferAddress;
	{
		VkBufferCreateInfo bufferInfo{
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = instances.size() * sizeof(VkAccelerationStructureInstanceKHR),
			.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
					 VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
		};
		VmaAllocationCreateInfo allocInfo{
			.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
				VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
			.usage = VMA_MEMORY_USAGE_AUTO
		};
		vulkan::checkResult(vmaCreateBuffer(context.allocator,
			&bufferInfo, &allocInfo,
			&instanceBuffer,
			&instanceBufferAllocation,
			&instanceBufferInfo));
		//get device address
		VkBufferDeviceAddressInfo addressInfo{
			.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
			.buffer = instanceBuffer
		};
		instanceBufferAddress = context.fnTable.vkGetBufferDeviceAddress(
			context.device, &addressInfo);
	}
	//type cast buffer
	std::span<VkAccelerationStructureInstanceKHR> asInstances{
		static_cast<VkAccelerationStructureInstanceKHR*>(instanceBufferInfo.pMappedData),
		instances.size()
	};

	//fill instance buffer
	auto pIn = instances.begin();
	auto pIdx = blasIdx.begin();
	auto pOut = asInstances.data();
	for (auto i = 0u; i < instances.size(); ++i, ++pIn, ++pIdx, ++pOut) {
		memcpy(pOut, &pIn->transform, sizeof(VkTransformMatrixKHR)); //copy transformation matrix
		pOut->instanceCustomIndex = pIn->customIndex;
		pOut->mask = pIn->mask;
		pOut->instanceShaderBindingTableRecordOffset = 0;
		pOut->flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
		pOut->accelerationStructureReference = blas[*pIdx].deviceAddress;
	}

	//Create tlas geometry
	VkAccelerationStructureGeometryKHR tlasGeometry{
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
		.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
		.flags = VK_GEOMETRY_OPAQUE_BIT_KHR
	};
	tlasGeometry.geometry.instances = VkAccelerationStructureGeometryInstancesDataKHR{
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
		.arrayOfPointers = VK_FALSE,
		.data = instanceBufferAddress
	};

	//get size info
	VkAccelerationStructureBuildGeometryInfoKHR tlasGeometryInfo{
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
		.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
		.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
		.geometryCount = 1,
		.pGeometries = &tlasGeometry
	};
	uint32_t primitiveCount = static_cast<uint32_t>(instances.size());
	VkAccelerationStructureBuildSizesInfoKHR tlasSizeInfo{
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
	};
	context.fnTable.vkGetAccelerationStructureBuildSizesKHR(
		context.device,
		VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
		&tlasGeometryInfo, &primitiveCount, &tlasSizeInfo);

	//create tlas
	vulkan::createAccelerationStructure(context, tlas, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, tlasSizeInfo);
	tlasGeometryInfo.dstAccelerationStructure = tlas.handle;

	//Create scratch buffer
	VkBuffer scratchBuffer;
	VmaAllocation scratchAllocation;
	VkDeviceAddress scratchAddress;
	{
		//allocate buffer
		VkBufferCreateInfo bufferInfo{
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size  = tlasSizeInfo.buildScratchSize,
			.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
		};
		VmaAllocationCreateInfo allocInfo{
			.usage = VMA_MEMORY_USAGE_AUTO
		};
		vulkan::checkResult(vmaCreateBuffer(context.allocator,
			&bufferInfo, &allocInfo,
			&scratchBuffer, &scratchAllocation, nullptr));
		//get device address
		VkBufferDeviceAddressInfo addressInfo{
			.sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
			.buffer = scratchBuffer
		};
		scratchAddress = context.fnTable.vkGetBufferDeviceAddress(context.device, &addressInfo);
	}
	//register scratch buffer
	tlasGeometryInfo.scratchData.deviceAddress = scratchAddress;

	//range info
	VkAccelerationStructureBuildRangeInfoKHR tlasRangeInfo{
		.primitiveCount = static_cast<uint32_t>(instances.size())
	};
	auto pRangeInfo = &tlasRangeInfo;

	//build tlas one device
	auto& con = context;
	vulkan::oneTimeSubmit(context, [&con, &tlasGeometryInfo, &pRangeInfo](VkCommandBuffer cmd) {
		con.fnTable.vkCmdBuildAccelerationStructuresKHR(cmd,
			1, &tlasGeometryInfo, &pRangeInfo);
	});

	//destroy buffers
	vmaDestroyBuffer(context.allocator, scratchBuffer, scratchAllocation);
	vmaDestroyBuffer(context.allocator, instanceBuffer, instanceBufferAllocation);
}

AccelerationStructure::Parameter::Parameter(
	vulkan::Context& context,
	std::span<const GeometryInstance> instances)
	: context(context)
{
	//collect unique geometries
	std::vector<const Geometry*> pGeometries{};
	std::vector<size_t> geometryIdx{};
	for (auto& instance : instances) {
		auto pCand = &instance.geometry.get();
		auto pGeo = std::find(pGeometries.begin(), pGeometries.end(), pCand);
		if (pGeo == pGeometries.end()) {
			geometryIdx.push_back(pGeometries.size());
			pGeometries.push_back(pCand);
		}
		else {
			geometryIdx.push_back(std::distance(pGeometries.begin(), pGeo));
		}
	}

	//build blas
	for (auto g : pGeometries)
		blas.push_back(createBlas(*g));

	//build tlas
	createTlas(instances, geometryIdx);

	//create descriptor info
	descriptorInfo = VkWriteDescriptorSetAccelerationStructureKHR{
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
		.accelerationStructureCount = 1,
		.pAccelerationStructures = &tlas.handle
	};
}

AccelerationStructure::Parameter::~Parameter() {
	//destroy all acceleration structures
	for (auto& b : blas)
		vulkan::destroyAccelerationStructure(context, b);
	vulkan::destroyAccelerationStructure(context, tlas);
}

void AccelerationStructure::bindParameter(VkWriteDescriptorSet& binding) const {
	binding.pNext = &param->descriptorInfo;
	binding.pBufferInfo = nullptr;
	binding.pImageInfo = nullptr;
	binding.pTexelBufferView = nullptr;
}

AccelerationStructure::AccelerationStructure(AccelerationStructure&&) noexcept = default;
AccelerationStructure& AccelerationStructure::operator=(AccelerationStructure&&) noexcept = default;

AccelerationStructure::AccelerationStructure(ContextHandle context, const GeometryInstance& instance)
	: AccelerationStructure(std::move(context), { &instance, 1 })
{}
AccelerationStructure::AccelerationStructure(ContextHandle context, std::span<const GeometryInstance> instances)
	: Resource(std::move(context))
	, param(std::make_unique<Parameter>(*getContext(), instances))
{}
AccelerationStructure::~AccelerationStructure() = default;

}
