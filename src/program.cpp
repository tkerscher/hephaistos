#include "hephaistos/program.hpp"

#include <cstdlib>
#include <stdexcept>

#include "volk.h"
#include "spirv_reflect.h"

#include "vk/result.hpp"
#include "vk/types.hpp"

namespace hephaistos {

/********************************** SUBGROUP **********************************/

namespace {

SubgroupProperties getSubgroupProperties(VkPhysicalDevice device) {
	//fetch subgroup properties
	VkPhysicalDeviceSubgroupProperties subgroupProps{
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES
	};
	VkPhysicalDeviceProperties2 props{
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
		.pNext = &subgroupProps
	};
	vkGetPhysicalDeviceProperties2(device, &props);

	//build struct
	return SubgroupProperties{
		.subgroupSize            = subgroupProps.subgroupSize,
		.basicSupport            = !!(subgroupProps.supportedOperations & VK_SUBGROUP_FEATURE_BASIC_BIT),
		.voteSupport             = !!(subgroupProps.supportedOperations & VK_SUBGROUP_FEATURE_VOTE_BIT),
		.arithmeticSupport       = !!(subgroupProps.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT),
		.ballotSupport           = !!(subgroupProps.supportedOperations & VK_SUBGROUP_FEATURE_BALLOT_BIT),
		.shuffleSupport          = !!(subgroupProps.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT),
		.shuffleRelativeSupport  = !!(subgroupProps.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT),
		.shuffleClusteredSupport = !!(subgroupProps.supportedOperations & VK_SUBGROUP_FEATURE_CLUSTERED_BIT),
		.quadSupport             = !!(subgroupProps.supportedOperations & VK_SUBGROUP_FEATURE_QUAD_BIT)
	};
}

}

SubgroupProperties getSubgroupProperties(const DeviceHandle& device) {
	return getSubgroupProperties(device->device);
}
SubgroupProperties getSubgroupProperties(const ContextHandle& context) {
	return getSubgroupProperties(context->physicalDevice);
}

/********************************** DISPATCH **********************************/

namespace vulkan {

struct Program {
	VkDescriptorSetLayout descriptorSetLayout = nullptr;
	VkPipelineLayout pipeLayout = nullptr;
	VkShaderModule shader = nullptr;
	VkPipeline pipeline = nullptr;
	uint32_t set = 0;

	std::vector<VkWriteDescriptorSet> boundParams;

	const Context& context;

	Program(const Context& context)
		: context(context)
	{}
};

}

void DispatchCommand::record(vulkan::Command& cmd) const {
	auto& prog = program.get();
	auto& context = prog.context;

	//we're working in the compute stage
	cmd.stage |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

	//bind pipeline
	context.fnTable.vkCmdBindPipeline(cmd.buffer,
		VK_PIPELINE_BIND_POINT_COMPUTE,
		prog.pipeline);

	//bind params
	context.fnTable.vkCmdPushDescriptorSetKHR(cmd.buffer,
		VK_PIPELINE_BIND_POINT_COMPUTE,
		prog.pipeLayout,
		prog.set,
		static_cast<uint32_t>(prog.boundParams.size()),
		prog.boundParams.data());

	//push constant if there is any
	if (!pushData.empty()) {
		context.fnTable.vkCmdPushConstants(cmd.buffer,
			prog.pipeLayout,
			VK_SHADER_STAGE_COMPUTE_BIT,
			0,
			static_cast<uint32_t>(pushData.size_bytes()),
			pushData.data());
	}

	//dispatch
	context.fnTable.vkCmdDispatch(cmd.buffer,
		groupCountX, groupCountY, groupCountZ);
}

DispatchCommand::DispatchCommand(const DispatchCommand&) = default;
DispatchCommand& DispatchCommand::operator=(const DispatchCommand&) = default;

DispatchCommand::DispatchCommand(DispatchCommand&&) noexcept = default;
DispatchCommand& DispatchCommand::operator=(DispatchCommand&&) noexcept = default;

DispatchCommand::DispatchCommand(
	const vulkan::Program& program,
	uint32_t x, uint32_t y, uint32_t z,
	std::span<const std::byte> push
)
	: groupCountX(x)
	, groupCountY(y)
	, groupCountZ(z)
	, pushData(push)
	, program(std::cref(program))
{}
DispatchCommand::~DispatchCommand() = default;

/*********************************** PROGRAM **********************************/

namespace {

void spvCheckResult(SpvReflectResult result) {
	if (result != SPV_REFLECT_RESULT_SUCCESS)
		throw std::runtime_error("Failed to retrieve reflection information from the shader!");
}

}

VkWriteDescriptorSet& Program::getBinding(uint32_t i) {
	if (i >= program->boundParams.size())
		throw std::runtime_error("There is no binding point at specified number!");

	return program->boundParams[i];
}

DispatchCommand Program::dispatch(std::span<const std::byte> push, uint32_t x, uint32_t y, uint32_t z) const {
	return DispatchCommand(*program, x, y, z, push);
}
DispatchCommand Program::dispatch(uint32_t x, uint32_t y, uint32_t z) const {
	return dispatch({}, x, y, z);
}

Program::Program(Program&&) noexcept = default;
Program& Program::operator=(Program&&) noexcept = default;

Program::Program(ContextHandle context, std::span<const uint32_t> code, std::span<const std::byte> specialization)
	: Resource(std::move(context))
	, program(std::make_unique<vulkan::Program>(*getContext()))
{
	//context reference
	auto& con = getContext();

	//create reflection module
	SpvReflectShaderModule reflectModule;
	spvCheckResult(spvReflectCreateShaderModule2(
		SPV_REFLECT_MODULE_FLAG_NO_COPY,
		code.size_bytes(), code.data(),
		&reflectModule));

	//create pipeline layout; we fill this as we gather more info
	VkPipelineLayoutCreateInfo layoutInfo{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
	};

	//We only support a single descriptor set
	if (reflectModule.descriptor_set_count > 1)
		throw std::logic_error("Programs are only allowed to have a single descriptor set!");
	//build descriptor set layout
	if (reflectModule.descriptor_set_count == 1) {
		auto& set = reflectModule.descriptor_sets[0];

		//reserve binding slots
		program->boundParams = std::vector<VkWriteDescriptorSet>(set.binding_count);

		//Create bindings
		auto pBinding = *set.bindings;
		std::vector<VkDescriptorSetLayoutBinding> bindings(set.binding_count);
		for (auto i = 0u; i < set.binding_count; ++i, ++pBinding) {
			bindings[i] = VkDescriptorSetLayoutBinding{
				.binding         = pBinding->binding,
				.descriptorType  = static_cast<VkDescriptorType>(pBinding->descriptor_type),
				.descriptorCount = pBinding->count,
				.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT
			};
			program->boundParams[i] = VkWriteDescriptorSet{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstBinding = pBinding->binding,
				.descriptorCount = 1,
				.descriptorType = static_cast<VkDescriptorType>(pBinding->descriptor_type)
			};
		}

		//Create set layout
		VkDescriptorSetLayoutCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
			.bindingCount = set.binding_count,
			.pBindings    = bindings.data()
		};
		vulkan::checkResult(con->fnTable.vkCreateDescriptorSetLayout(
			con->device, &info, nullptr, &program->descriptorSetLayout));

		//register in pipeline layout
		layoutInfo.setLayoutCount = 1;
		layoutInfo.pSetLayouts = &program->descriptorSetLayout;
	}

	//read push descriptor
	VkPushConstantRange push{};
	if (reflectModule.push_constant_block_count > 1)
		throw std::logic_error("Multiple push constant found, but only up to one is supported!");
	if (reflectModule.push_constant_block_count == 1) {
		//read push size
		push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		push.size = reflectModule.push_constant_blocks->size;

		//register in layout
		layoutInfo.pushConstantRangeCount = 1;
		layoutInfo.pPushConstantRanges = &push;
	}

	//Create pipeline layout
	vulkan::checkResult(con->fnTable.vkCreatePipelineLayout(
		con->device, &layoutInfo, nullptr, &program->pipeLayout));

	//read specialization constants map
	auto specSlots = std::min(
		reflectModule.specialization_constant_count,
		static_cast<uint32_t>(specialization.size_bytes() / 4));
	std::vector<VkSpecializationMapEntry> specMap(specSlots);
	for (auto i = 0u; i < specSlots; ++i) {
		//all types allowed for specialization are 4 bytes long
		//we assume the spec const to be tightly packed
		specMap[i] = VkSpecializationMapEntry{
			.constantID = reflectModule.specialization_constants[i].constant_id,
			.offset = 4 * i,
			.size = 4
		};
	}

	//we're done reflecting
	spvReflectDestroyShaderModule(&reflectModule);

	//build specialization info
	VkSpecializationInfo specInfo{
		.mapEntryCount = specSlots,
		.pMapEntries   = specMap.data(),
		.dataSize      = specialization.size_bytes(),
		.pData         = specialization.data()
	};

	//compile code into shader module
	VkShaderModuleCreateInfo shaderInfo{
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = code.size_bytes(),
		.pCode = code.data()
	};
	vulkan::checkResult(con->fnTable.vkCreateShaderModule(
		con->device, &shaderInfo, nullptr, &program->shader));

	//Shader stage create info
	VkPipelineShaderStageCreateInfo stageInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_COMPUTE_BIT,
			.module = program->shader,
			.pName = reflectModule.entry_point_name,
			.pSpecializationInfo = specMap.empty() ? nullptr : &specInfo
	};

	//create compute pipeline
	VkComputePipelineCreateInfo pipeInfo{
		.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
		.stage = stageInfo,
		.layout = program->pipeLayout,
	};
	vulkan::checkResult(con->fnTable.vkCreateComputePipelines(
		con->device, con->cache,
		1, &pipeInfo,
		nullptr, &program->pipeline));
}
Program::Program(ContextHandle context, std::span<const uint32_t> code)
	: Program(std::move(context), code, {})
{}
Program::~Program() {
	auto& context = getContext();
	if (program) {
		context->fnTable.vkDestroyPipeline(context->device, program->pipeline, nullptr);
		context->fnTable.vkDestroyShaderModule(context->device, program->shader, nullptr);
		context->fnTable.vkDestroyPipelineLayout(context->device, program->pipeLayout, nullptr);
		context->fnTable.vkDestroyDescriptorSetLayout(context->device, program->descriptorSetLayout, nullptr);
	}
}

}
