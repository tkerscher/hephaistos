#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "hephaistos/config.hpp"

namespace hephaistos {

class HEPHAISTOS_API Compiler {
public:
	//heterogenous string hash
	struct string_hash {
		using hash_type = std::hash<std::string_view>;
		using is_transparent = void;

		[[nodiscard]] std::size_t operator()(const char* str) const {
			return hash_type{}(str);
		}
		[[nodiscard]] std::size_t operator()(std::string_view str) const {
			return hash_type{}(str);
		}
		[[nodiscard]] std::size_t operator()(const std::string& str) const {
			return hash_type{}(str);
		}
	};

	using HeaderMap = std::unordered_map<std::string, std::string, string_hash, std::equal_to<>>;

public:

	void addIncludeDir(std::filesystem::path dir);
	void popIncludeDir();
	void clearIncludeDir();

	[[nodiscard]] std::vector<uint32_t> compile(std::string_view code) const;
	[[nodiscard]] std::vector<uint32_t> compile(std::string_view code, const HeaderMap& headers) const;

	Compiler& operator=(Compiler&&) noexcept;
	Compiler(Compiler&&) noexcept;

	Compiler& operator=(const Compiler&) = delete;
	Compiler(const Compiler&) = delete;

	Compiler();
	~Compiler();

private:
	std::vector<std::filesystem::path> includeDirs;
};

}
