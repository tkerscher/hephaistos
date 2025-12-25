#include "hephaistos/raytracing/pipeline.hpp"

#include "vk/types.hpp"
#include "vk/util.hpp"

#include "raytracing/extension.hpp"

#include "vk/reflection.hpp"

namespace hephaistos {

namespace {

//all ray tracing shader stages
auto constexpr RAYTRACING_SHADER_STAGES =
	VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
	VK_SHADER_STAGE_CALLABLE_BIT_KHR |
	VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
	VK_SHADER_STAGE_INTERSECTION_BIT_KHR |
	VK_SHADER_STAGE_MISS_BIT_KHR |
	VK_SHADER_STAGE_RAYGEN_BIT_KHR;

VkStridedDeviceAddressRegionKHR toAddressRegion(ShaderBindingTableRegion region) {
	return {
		.deviceAddress = region.address,
		.stride = region.stride,
		.size = region.stride * region.count
	};
}

}

/**************************** SHADER BINDING TABLE ****************************/

ShaderBindingTable::operator ShaderBindingTableRegion() const {
	return region;
}

void ShaderBindingTable::onDestroy() {
	buffer.reset();
	region = {};
}

ShaderBindingTable::ShaderBindingTable(
	ContextHandle context, BufferHandle buffer,
	uint32_t stride, uint32_t count
)
	: Resource(std::move(context))
	, buffer(std::move(buffer))
	, region({ vulkan::getBufferDeviceAddress(this->buffer), stride, count })
{}
ShaderBindingTable::~ShaderBindingTable() = default;

/**************************** RAY TRACING PIPELINE ****************************/

namespace vulkan {

struct RayTracingPipeline {
	VkDescriptorSetLayout descriptorSetLayout = nullptr;
	VkPipelineLayout layout = nullptr;
	VkPipeline pipeline = nullptr;

	std::vector<std::byte> shaderHandleStorage = {};

	const Context& context;

	RayTracingPipeline(const Context& context)
		: context(context)
	{}
};

}

namespace {

auto constexpr SBT_BUFFER_USAGE_FLAGS =
	VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR |
	VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

auto constexpr SBT_VMA_USAGE_FLAGS =
	VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
	VMA_ALLOCATION_CREATE_MAPPED_BIT;

struct PipelineBuilder {
	vulkan::LayoutReflectionBuilder reflection;

	std::vector<VkShaderModuleCreateInfo> modules;
	std::vector<VkPipelineShaderStageCreateInfo> stages;
	std::vector<VkRayTracingShaderGroupCreateInfoKHR> groups;

	VkSpecializationInfo specInfo;

	void operator()(const RayGenerateShader& shader);
	void operator()(const RayMissShader& shader);
	void operator()(const RayHitShader& shader);
	void operator()(const CallableShader& shader);

	void addShader(const ShaderCode& shader, VkShaderStageFlagBits stage);

	void finalize(std::span<const std::byte> specialization);
};

void PipelineBuilder::operator()(const RayGenerateShader& shader) {
	reflection.add(shader.code.code);
	groups.push_back(VkRayTracingShaderGroupCreateInfoKHR{
		.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
		.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
		.generalShader = static_cast<uint32_t>(stages.size()),
		.closestHitShader = VK_SHADER_UNUSED_KHR,
		.anyHitShader = VK_SHADER_UNUSED_KHR,
		.intersectionShader = VK_SHADER_UNUSED_KHR
	});
	addShader(shader.code, VK_SHADER_STAGE_RAYGEN_BIT_KHR);
}
void PipelineBuilder::operator()(const RayMissShader& shader) {
	reflection.add(shader.code.code);
	groups.push_back(VkRayTracingShaderGroupCreateInfoKHR{
		.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
		.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
		.generalShader = static_cast<uint32_t>(stages.size()),
		.closestHitShader = VK_SHADER_UNUSED_KHR,
		.anyHitShader = VK_SHADER_UNUSED_KHR,
		.intersectionShader = VK_SHADER_UNUSED_KHR
	});
	addShader(shader.code, VK_SHADER_STAGE_MISS_BIT_KHR);
}
void PipelineBuilder::operator()(const RayHitShader& shader) {
	reflection.add(shader.closest.code);
	groups.push_back(VkRayTracingShaderGroupCreateInfoKHR{
		.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
		.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
		.generalShader = VK_SHADER_UNUSED_KHR,
		.closestHitShader = static_cast<uint32_t>(stages.size()),
		.anyHitShader = VK_SHADER_UNUSED_KHR,
		.intersectionShader = VK_SHADER_UNUSED_KHR
	});
	addShader(shader.closest, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);

	if (!shader.any.code.empty()) {
		reflection.add(shader.any.code);
		groups.back().anyHitShader = static_cast<uint32_t>(stages.size());
		addShader(shader.any, VK_SHADER_STAGE_ANY_HIT_BIT_KHR);
	}
}
void PipelineBuilder::operator()(const CallableShader& shader) {
	reflection.add(shader.code.code);
	groups.push_back(VkRayTracingShaderGroupCreateInfoKHR{
		.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
		.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
		.generalShader = static_cast<uint32_t>(stages.size()),
		.closestHitShader = VK_SHADER_UNUSED_KHR,
		.anyHitShader = VK_SHADER_UNUSED_KHR,
		.intersectionShader = VK_SHADER_UNUSED_KHR
	});
	addShader(shader.code, VK_SHADER_STAGE_CALLABLE_BIT_KHR);
}

void PipelineBuilder::addShader(const ShaderCode& shader, VkShaderStageFlagBits stage) {
	modules.push_back(VkShaderModuleCreateInfo{
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = shader.code.size_bytes(),
		.pCode = shader.code.data()
	});
	stages.push_back(VkPipelineShaderStageCreateInfo{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
		.stage = stage,
		.pName = shader.entryName.data()
	});
}

void PipelineBuilder::finalize(std::span<const std::byte> specialization) {
	//fill in specialization & link shader modules
	auto specMap = reflection.createSpecializationMap(specialization.size_bytes() / 4);
	specInfo = vulkan::createSpecializationInfo(specMap, specialization);
	for (auto i = 0; i < stages.size(); ++i) {
		stages[i].pNext = &modules[i];
		stages[i].pSpecializationInfo = &specInfo;
	}
}

inline uint32_t align(uint32_t value, uint32_t alignment) {
	return (value + alignment - 1) & ~(alignment - 1);
}

}

ShaderBindingTable RayTracingPipeline::createShaderBindingTable(
	uint32_t firstGroupIdx, uint32_t count
) {
	//range check
	if (firstGroupIdx + count >= handleCount)
		throw std::runtime_error("Requested range outside available shader range");

	//fetch alignments
	auto ext = getExtension<RayTracingExtension>(*getContext(), "RayTracing");
	auto baseAlign = ext->shaderGroupProperties.shaderGroupBaseAlignment;
	auto handleAlign = ext->shaderGroupProperties.shaderGroupHandleAlignment;
	auto handleSize = ext->shaderGroupProperties.shaderGroupHandleSize;
	auto entrySize = align(handleSize, handleAlign);
	auto padding = entrySize - handleSize;
	//create and fill sbt buffer
	auto buffer = vulkan::createBufferAligned(
		getContext(),
		entrySize * count,
		baseAlign,
		SBT_BUFFER_USAGE_FLAGS,
		SBT_VMA_USAGE_FLAGS
	);
	auto pData = static_cast<std::byte*>(buffer->allocInfo.pMappedData);
	auto pStorage = handleStorage.data();
	pStorage += handleSize * firstGroupIdx;
	for (auto i = 0; i < count; ++i) {
		std::memcpy(pData, pStorage, handleSize);
		pData += entrySize;
		std::memset(pData, 0, padding);
		pData += padding;
		pStorage += handleSize;
	}
	//return SBT
	return ShaderBindingTable(getContext(), std::move(buffer), entrySize, count);
}

ShaderBindingTable RayTracingPipeline::createShaderBindingTable(
	std::span<const ShaderBindingTableEntry> entries
) {
	//we only have one common stride for our SBT
	//it will be determined by the largest block of data we want to store
	//in between, rounded up to alignment
	
	//get larget block of data
	size_t maxRecordSize = 0;
	for (auto& entry : entries)
		maxRecordSize = std::max(maxRecordSize, entry.shaderRecord.size_bytes());
	//fetch alignments
	auto ext = getExtension<RayTracingExtension>(*getContext(), "RayTracing");
	auto baseAlign = ext->shaderGroupProperties.shaderGroupBaseAlignment;
	auto handleAlign = ext->shaderGroupProperties.shaderGroupHandleAlignment;
	auto handleSize = ext->shaderGroupProperties.shaderGroupHandleSize;
	auto entrySize = align(handleSize + maxRecordSize, handleAlign);
	auto entryCount = static_cast<uint32_t>(entries.size());

	//create buffer
	auto buffer = vulkan::createBufferAligned(
		getContext(),
		entrySize * entryCount,
		baseAlign,
		SBT_BUFFER_USAGE_FLAGS,
		SBT_VMA_USAGE_FLAGS
	);
	//fill buffer with entries
	auto pData = static_cast<std::byte*>(buffer->allocInfo.pMappedData);
	auto pStorage = handleStorage.data();
	for (auto& entry : entries) {
		//process entry
		if (entry.groupIndex == ~0u) //null handle?
			std::memset(pData, 0, handleSize);
		else if (entry.groupIndex >= handleStorage.size())
			throw std::runtime_error("Referenced group out of range");
		else
			std::memcpy(pData, pStorage + (entry.groupIndex * handleSize), handleSize);
		pData += handleSize;
		//copy data
		if (!entry.shaderRecord.empty()) {
			std::memcpy(pData, entry.shaderRecord.data(), entry.shaderRecord.size_bytes());
			pData += entry.shaderRecord.size_bytes();
		}
		//pad with zeros
		auto padding = entrySize - handleSize - entry.shaderRecord.size_bytes();
		std::memset(pData, 0, padding);
		pData += padding;
	}

	//return SBT
	return ShaderBindingTable(getContext(), std::move(buffer), entrySize, entryCount);
}

void RayTracingPipeline::onDestroy() {
	if (pipeline) {
		auto& context = getContext();
		context->fnTable.vkDestroyPipeline(context->device, pipeline->pipeline, nullptr);
		context->fnTable.vkDestroyPipelineLayout(context->device, pipeline->layout, nullptr);
		context->fnTable.vkDestroyDescriptorSetLayout(context->device, pipeline->descriptorSetLayout, nullptr);
		
		pipeline.reset();
		bindingTraits.clear();
		boundParams.clear();

		handleStorage.clear();
		handleCount = 0;
	}
}

TraceRaysCommand RayTracingPipeline::traceRays(
	ShaderBindings bindings, RayCount rayCount, std::span<const std::byte> push
) const {
	checkAllBindingsBound();
	return TraceRaysCommand(*pipeline, boundParams, bindings, rayCount, push);
}

TraceRaysIndirectCommand RayTracingPipeline::traceRaysIndirect(
	ShaderBindings bindings, const Tensor<std::byte>& tensor, uint64_t offset
) const {
	checkAllBindingsBound();
	return TraceRaysIndirectCommand(*pipeline, boundParams, bindings, tensor, offset, {});
}
TraceRaysIndirectCommand RayTracingPipeline::traceRaysIndirect(
	ShaderBindings bindings,
	std::span<const std::byte> push,
	const Tensor<std::byte>& tensor,
	uint64_t offset
) const {
	checkAllBindingsBound();
	return TraceRaysIndirectCommand(*pipeline, boundParams, bindings, tensor, offset, push);
}

RayTracingPipeline::RayTracingPipeline(
	ContextHandle context,
	std::span<const RayTracingShader> shader,
	uint32_t maxRecursionDepth
)
	: RayTracingPipeline(std::move(context), shader, {}, maxRecursionDepth)
{}
RayTracingPipeline::RayTracingPipeline(
	ContextHandle context,
	std::span<const RayTracingShader> shaders,
	std::span<const std::byte> specialization,
	uint32_t maxRecursionDepth
)
	: Resource(std::move(context))
	, BindingTarget()
	, pipeline(std::make_unique<vulkan::RayTracingPipeline>(*getContext()))
{
	auto& con = *getContext();
	auto ext = getExtension<RayTracingExtension>(*getContext(), "RayTracing");

	//check limits
	if (maxRecursionDepth > ext->props.maxRayRecursionDepth)
		throw std::runtime_error("Specified max recursion depth exceeds device limit");

	//process all shader
	PipelineBuilder builder;
	for (auto& shader : shaders)
		std::visit(builder, shader);
	builder.finalize(specialization);
	//move params
	bindingTraits = std::move(builder.reflection.traits);
	boundParams = std::move(builder.reflection.params);

	//create pipeline layout
	builder.reflection.createDescriptorSetLayout(con, &pipeline->descriptorSetLayout);
	builder.reflection.createPipelineLayout(con, &pipeline->descriptorSetLayout, &pipeline->layout);
	//create ray tracing pipeline
	VkRayTracingPipelineCreateInfoKHR pipeInfo{
		.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
		.stageCount = static_cast<uint32_t>(builder.stages.size()),
		.pStages = builder.stages.data(),
		.groupCount = static_cast<uint32_t>(builder.groups.size()),
		.pGroups = builder.groups.data(),
		.maxPipelineRayRecursionDepth = maxRecursionDepth,
		.layout = pipeline->layout
	};
	vulkan::checkResult(con.fnTable.vkCreateRayTracingPipelinesKHR(con.device,
		VK_NULL_HANDLE,
		con.cache,
		1, &pipeInfo,
		nullptr,
		&pipeline->pipeline));

	//fetch SBT group handles; these will be tightly packed
	handleSize = ext->shaderGroupProperties.shaderGroupHandleSize;
	handleCount = static_cast<uint32_t>(builder.groups.size());
	auto storageSize = handleSize * handleCount;
	handleStorage.resize(storageSize);
	vulkan::checkResult(con.fnTable.vkGetRayTracingShaderGroupHandlesKHR(
		con.device,
		pipeline->pipeline,
		0, handleCount,
		storageSize,
		handleStorage.data()
	));
}
RayTracingPipeline::~RayTracingPipeline() {
	onDestroy();
}

/****************************** DISPATCH COMMANDS *****************************/

void TraceRaysCommand::record(vulkan::Command& cmd) const {
	auto& prog = pipeline.get();
	auto& context = prog.context;

	//check limit
	auto ext = getExtension<RayTracingExtension>(context, "RayTracing");
	auto totalCount = rayCount.x * rayCount.y * rayCount.z;
	if (totalCount > ext->props.maxRayDispatchCount)
		throw std::runtime_error("Total ray count exceeds device limit");

	//we're working in the ray tracing stage
	cmd.stage |= VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;

	//bind pipeline
	context.fnTable.vkCmdBindPipeline(cmd.buffer,
		VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
		prog.pipeline);

	//bind params if there are any
	if (!params.empty()) {
		context.fnTable.vkCmdPushDescriptorSetKHR(cmd.buffer,
			VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
			prog.layout,
			0,
			static_cast<uint32_t>(params.size()),
			params.data());
	}
	//push constant if there is any
	if (!pushData.empty()) {
		context.fnTable.vkCmdPushConstants(cmd.buffer,
			prog.layout,
			RAYTRACING_SHADER_STAGES, //TODO: Check this
			0,
			static_cast<uint32_t>(pushData.size_bytes()),
			pushData.data());
	}

	//convert SBT pointers
	auto rayGen = toAddressRegion(shaderBindings.rayGenShaders);
	auto miss = toAddressRegion(shaderBindings.missShaders);
	auto hit = toAddressRegion(shaderBindings.hitShaders);
	auto callable = toAddressRegion(shaderBindings.callableShaders);
	//dispatch
	context.fnTable.vkCmdTraceRaysKHR(cmd.buffer,
		&rayGen, &miss, &hit, &callable,
		rayCount.x, rayCount.y, rayCount.z);
}

TraceRaysCommand::TraceRaysCommand(const TraceRaysCommand&) = default;
TraceRaysCommand& TraceRaysCommand::operator=(const TraceRaysCommand&) = default;

TraceRaysCommand::TraceRaysCommand(TraceRaysCommand&&) noexcept = default;
TraceRaysCommand& TraceRaysCommand::operator=(TraceRaysCommand&&) noexcept = default;

TraceRaysCommand::TraceRaysCommand(
	const vulkan::RayTracingPipeline& pipeline,
	std::vector<VkWriteDescriptorSet> params,
	ShaderBindings shaderBindings,
	RayCount rayCount,
	std::span<const std::byte> push
)
	: pipeline(std::cref(pipeline))
	, params(std::move(params))
	, shaderBindings(shaderBindings)
	, rayCount(rayCount)
	, pushData(push)
{}
TraceRaysCommand::~TraceRaysCommand() = default;

void TraceRaysIndirectCommand::record(vulkan::Command& cmd) const {
	auto buffer = tensor.get().getBuffer().buffer;
	auto& prog = pipeline.get();
	auto& context = prog.context;
	auto address = tensor.get().address() + offset;

	//we're working on the ray tracing pipeline
	cmd.stage |=
		VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT |
		VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;

	//bind pipeline
	context.fnTable.vkCmdBindPipeline(cmd.buffer,
		VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
		prog.pipeline);

	//bind params if there are any
	if (!params.empty()) {
		context.fnTable.vkCmdPushDescriptorSetKHR(cmd.buffer,
			VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
			prog.layout,
			0,
			static_cast<uint32_t>(params.size()),
			params.data());
	}
	//push constant if there is any
	if (!pushData.empty()) {
		context.fnTable.vkCmdPushConstants(cmd.buffer,
			prog.layout,
			RAYTRACING_SHADER_STAGES,
			0,
			static_cast<uint32_t>(pushData.size_bytes()),
			pushData.data());
	}

	//barrier to ensure indirect data is complete
	VkBufferMemoryBarrier barrier{
		.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
		.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
		.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
		.buffer = buffer,
		.offset = offset,
		.size = 12 // 3 * int
	};
	context.fnTable.vkCmdPipelineBarrier(cmd.buffer,
		VK_PIPELINE_STAGE_TRANSFER_BIT |
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
		VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
		VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0,
		0, nullptr,
		1, &barrier,
		0, nullptr);

	//convert SBT pointers
	auto rayGen = toAddressRegion(shaderBindings.rayGenShaders);
	auto miss = toAddressRegion(shaderBindings.missShaders);
	auto hit = toAddressRegion(shaderBindings.hitShaders);
	auto callable = toAddressRegion(shaderBindings.callableShaders);
	//dispatch indirect
	context.fnTable.vkCmdTraceRaysIndirectKHR(cmd.buffer,
		&rayGen, &miss, &hit, &callable,
		address);
}

TraceRaysIndirectCommand::TraceRaysIndirectCommand(const TraceRaysIndirectCommand&) = default;
TraceRaysIndirectCommand& TraceRaysIndirectCommand::operator=(const TraceRaysIndirectCommand&) = default;

TraceRaysIndirectCommand::TraceRaysIndirectCommand(TraceRaysIndirectCommand&&) noexcept = default;
TraceRaysIndirectCommand& TraceRaysIndirectCommand::operator=(TraceRaysIndirectCommand&&) noexcept = default;

TraceRaysIndirectCommand::TraceRaysIndirectCommand(
	const vulkan::RayTracingPipeline& pipeline,
	std::vector<VkWriteDescriptorSet> params,
	ShaderBindings bindings,
	const Tensor<std::byte>& tensor,
	uint64_t offset,
	std::span<const std::byte> push
)
	: pipeline(pipeline)
	, params(std::move(params))
	, shaderBindings(bindings)
	, tensor(std::cref(tensor))
	, offset(offset)
	, pushData(push)
{}
TraceRaysIndirectCommand::~TraceRaysIndirectCommand() = default;

}
