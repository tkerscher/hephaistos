#include "hephaistos/stopwatch.hpp"

#include <limits>

#include "volk.h"

#include "vk/result.hpp"
#include "vk/types.hpp"

namespace hephaistos {

namespace {

class TimeStampCommand : public Command {
public:
	void record(vulkan::Command& cmd) const override {
		//TODO: really necessary?
		cmd.stage |= stage;

		context.fnTable.vkCmdWriteTimestamp(cmd.buffer,
			stage, queryPool, query);
	}

	TimeStampCommand(
		const vulkan::Context& context,
		VkPipelineStageFlagBits stage,
		VkQueryPool queryPool,
		uint32_t query)
		: context(context)
		, stage(stage)
		, queryPool(queryPool)
		, query(query)
	{}
	virtual ~TimeStampCommand() = default;

private:
	const vulkan::Context& context;
	VkPipelineStageFlagBits stage;
	VkQueryPool queryPool;
	uint32_t query;
};

}

struct StopWatch::pImp {
	const vulkan::Context& context;

	VkQueryPool queryPool;
	uint32_t validBits;
	float period;

	uint32_t count;
	std::vector<TimeStampCommand> commands;

	pImp(const vulkan::Context& context, uint32_t count)
		: context(context)
		, queryPool(nullptr)
		, count(count)
	{
		//query timestamp period
		VkPhysicalDeviceProperties devProps;
		vkGetPhysicalDeviceProperties(context.physicalDevice, &devProps);
		period = devProps.limits.timestampPeriod;

		//query timestamp valid bits
		uint32_t n;
		vkGetPhysicalDeviceQueueFamilyProperties(context.physicalDevice, &n, nullptr);
		std::vector<VkQueueFamilyProperties> qProps(n);
		vkGetPhysicalDeviceQueueFamilyProperties(context.physicalDevice, &n, qProps.data());
		validBits = qProps[context.queueFamily].timestampValidBits;

		//create query pool
		VkQueryPoolCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
			.queryType = VK_QUERY_TYPE_TIMESTAMP,
			.queryCount = count
		};
		vulkan::checkResult(context.fnTable.vkCreateQueryPool(
			context.device, &info, nullptr, &queryPool));

		//before use we have to reset it once
		context.fnTable.vkResetQueryPool(context.device, queryPool, 0, count);

		//create commands
		commands.emplace_back(context, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, 0);
		for (auto i = 1u; i < count; ++i)
			commands.emplace_back(context, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, i);
	}
	~pImp() {
		context.fnTable.vkDestroyQueryPool(
			context.device, queryPool, nullptr);
	}
};

const Command& StopWatch::start() {
	return _pImp->commands.at(0);
}
const Command& StopWatch::stop(uint32_t id) {
	return _pImp->commands.at(id + 1);
}

void StopWatch::reset() {
	_pImp->context.fnTable.vkResetQueryPool(
		_pImp->context.device,
		_pImp->queryPool,
		0,
		_pImp->count);
}
std::vector<double> StopWatch::getTimeStamps(bool wait) const {
	VkQueryResultFlags flags =
		VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT;
	if (wait)
		flags |= VK_QUERY_RESULT_WAIT_BIT;

	struct Result {
		uint64_t timestamp;
		uint64_t valid;
	};
	std::vector<Result> queryResults(_pImp->count);

	auto result = _pImp->context.fnTable.vkGetQueryPoolResults(
		_pImp->context.device,
		_pImp->queryPool,
		0, _pImp->count,
		sizeof(Result) * _pImp->count,
		queryResults.data(),
		sizeof(Result),
		flags);
	if (result < 0)
		vulkan::checkResult(result); //will throw

	std::vector<double> timestamps(_pImp->count);
	for (auto i = 0u; i < _pImp->count; ++i) {
		auto& result = queryResults[i];
		if (result.valid) {
			result.timestamp >>= (64 - _pImp->validBits);
			timestamps[i] = result.timestamp * static_cast<double>(_pImp->period);
		}
		else {
			timestamps[i] = std::numeric_limits<double>::quiet_NaN();
		}
	}

	return timestamps;
}

StopWatch::StopWatch(StopWatch&& other) noexcept = default;
StopWatch& StopWatch::operator=(StopWatch&& other) noexcept = default;

StopWatch::StopWatch(ContextHandle context, uint32_t stops)
	: Resource(std::move(context))
	, _pImp(std::make_unique<pImp>(*getContext(), stops + 1))
{}
StopWatch::~StopWatch() = default;

}
