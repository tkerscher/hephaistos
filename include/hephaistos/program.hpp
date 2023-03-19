#pragma once

#include <span>
#include <type_traits>
#include <vector>

#include "hephaistos/command.hpp"

//fwd
struct VkWriteDescriptorSet;

namespace hephaistos {

namespace vulkan {
	struct Param;
	struct Program;
}

struct SubgroupProperties {
	uint32_t subgroupSize;
	bool basicSupport;
	bool voteSupport;
	bool arithmeticSupport;
	bool ballotSupport;
	bool shuffleSupport;
	bool shuffleRelativeSupport;
	bool shuffleClusteredSupport;
	bool quadSupport;
};
HEPHAISTOS_API SubgroupProperties getSubgroupProperties(const DeviceHandle& device);
HEPHAISTOS_API SubgroupProperties getSubgroupProperties(const ContextHandle& context);

class HEPHAISTOS_API DispatchCommand : public Command {
public:
	uint32_t groupCountX;
	uint32_t groupCountY;
	uint32_t groupCountZ;
	std::span<const std::byte> pushData;

	virtual void record(vulkan::Command& cmd) const override;

	DispatchCommand(const DispatchCommand&);
	DispatchCommand& operator=(const DispatchCommand&);

	DispatchCommand(DispatchCommand&&) noexcept;
	DispatchCommand& operator=(DispatchCommand&&) noexcept;

	DispatchCommand(const vulkan::Program& program, uint32_t x, uint32_t y, uint32_t z, std::span<const std::byte> push);
	virtual ~DispatchCommand();

private:
	std::reference_wrapper<const vulkan::Program> program;
};

class HEPHAISTOS_API Program : public Resource {
public:
	VkWriteDescriptorSet& getBinding(uint32_t i);

	template<class T>
	void bindParameter(const T& param, uint32_t binding) {
		param.bindParameter(getBinding(binding));
	}
	template<class ...T>
	void bindParameterList(const T&...param) {
		uint32_t binding = 0;
		(param.bindParameter(getBinding(binding++)),...);
	}

	DispatchCommand dispatch(std::span<const std::byte> push, uint32_t x = 1, uint32_t y = 1, uint32_t z = 1) const;
	DispatchCommand dispatch(uint32_t x = 1, uint32_t y = 1, uint32_t z = 1) const;
	template<class T, typename = typename std::enable_if_t<std::is_standard_layout_v<T> && !std::is_integral_v<T>>>
	DispatchCommand dispatch(const T& push, uint32_t x = 1, uint32_t y = 1, uint32_t z = 1) const {
		return dispatch({ reinterpret_cast<const std::byte*>(&push), sizeof(T)}, x, y, z);
	}

	Program(const Program&) = delete;
	Program& operator=(const Program&) = delete;

	Program(Program&& other) noexcept;
	Program& operator=(Program&& other) noexcept;

	Program(ContextHandle context, std::span<const uint32_t> code);
	Program(ContextHandle context, std::span<const uint32_t> code, std::span<const std::byte> specialization);
	template<class T>
	Program(ContextHandle context, std::span<const uint32_t> code, const T& specialization)
		: Program(std::move(context), code, std::as_bytes(std::span<const T>{ &specialization, 1 }))
	{}
	~Program();

private:
	std::unique_ptr<vulkan::Program> program;
};

class HEPHAISTOS_API FlushMemoryCommand : public Command{
public:
	//The command on its own cant do anything so we're save to
	//just store the reference instead
	std::reference_wrapper<const vulkan::Context> context;

	virtual void record(vulkan::Command& cmd) const override;

	FlushMemoryCommand(const FlushMemoryCommand&);
	FlushMemoryCommand& operator=(const FlushMemoryCommand&);

	FlushMemoryCommand(const ContextHandle& context);
	virtual ~FlushMemoryCommand();
};
[[nodiscard]] inline FlushMemoryCommand flushMemory(const ContextHandle& context) {
	return FlushMemoryCommand(context);
}

}
