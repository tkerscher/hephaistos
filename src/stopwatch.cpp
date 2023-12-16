#include "hephaistos/stopwatch.hpp"

#include <array>
#include <limits>

#include "volk.h"

#include "vk/result.hpp"
#include "vk/types.hpp"

namespace hephaistos {

namespace {

class TimeStampCommand : public Command {
public:
    const vulkan::Context& context;
    VkPipelineStageFlagBits stage;
    VkQueryPool queryPool;
    uint32_t query;

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
};

}

struct StopWatch::pImp {
    const vulkan::Context& context;

    VkQueryPool queryPool;
    uint32_t validBits;
    float period;

    TimeStampCommand startCommand;
    TimeStampCommand endCommand;

    pImp(const vulkan::Context& context)
        : context(context)
        , queryPool(nullptr)
        , startCommand(context, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, nullptr, 0)
        , endCommand(context, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, nullptr, 1)
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
            .queryCount = 2
        };
        vulkan::checkResult(context.fnTable.vkCreateQueryPool(
            context.device, &info, nullptr, &queryPool));

        //before use we have to reset it once
        context.fnTable.vkResetQueryPool(context.device, queryPool, 0, 2);

        //register query pool in commands
        startCommand.queryPool = queryPool;
        endCommand.queryPool = queryPool;
    }
    ~pImp() {
        context.fnTable.vkDestroyQueryPool(
            context.device, queryPool, nullptr);
    }
};

const Command& StopWatch::start() {
    return _pImp->startCommand;
}
const Command& StopWatch::stop() {
    return _pImp->endCommand;
}

void StopWatch::reset() {
    _pImp->context.fnTable.vkResetQueryPool(
        _pImp->context.device,
        _pImp->queryPool,
        0, 2);

    //reset counter
    _pImp->endCommand.query = 1;
}

double StopWatch::getElapsedTime(bool wait) const {
    VkQueryResultFlags flags =
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT;
    if (wait)
        flags |= VK_QUERY_RESULT_WAIT_BIT;

    struct Result {
        uint64_t timestamp;
        uint64_t valid;
    };
    std::array<Result, 2> queries;

    auto result = _pImp->context.fnTable.vkGetQueryPoolResults(
        _pImp->context.device,
        _pImp->queryPool,
        0, 2,
        2 * sizeof(Result),
        queries.data(),
        sizeof(Result),
        flags);
    if (result < 0)
        vulkan::checkResult(result); //will throw
    
    if (queries[0].valid && queries[1].valid) {
        auto delta = queries[1].timestamp - queries[0].timestamp;
        delta >>= (64 - _pImp->validBits);
        return delta * static_cast<double>(_pImp->period);
    }
    else {
        return std::numeric_limits<double>::quiet_NaN();
    }
}

StopWatch::StopWatch(StopWatch&& other) noexcept = default;
StopWatch& StopWatch::operator=(StopWatch&& other) noexcept = default;

StopWatch::StopWatch(ContextHandle context)
    : Resource(std::move(context))
    , _pImp(std::make_unique<pImp>(*getContext()))
{}
StopWatch::~StopWatch() = default;

}
