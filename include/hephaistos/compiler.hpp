#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "hephaistos/config.hpp"

namespace hephaistos {

class HEPHAISTOS_API Compiler {
public:
	using HeaderMap = std::unordered_map<std::string, std::string>;

	std::vector<uint32_t> compile(std::string_view code) const;
	std::vector<uint32_t> compile(std::string_view code, const HeaderMap& headers) const;

	Compiler& operator=(Compiler&&) noexcept;
	Compiler(Compiler&&) noexcept;

	Compiler& operator=(const Compiler&) = delete;
	Compiler(const Compiler&) = delete;

	Compiler();
	~Compiler();
};

}
