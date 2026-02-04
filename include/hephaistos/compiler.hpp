#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "hephaistos/config.hpp"

namespace hephaistos {

namespace detail {

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

}

/**
 * @brief Maps include paths to source code
*/
using HeaderMap = std::unordered_map<std::string, std::string, detail::string_hash, std::equal_to<>>;

/**
 * @brief Enumeration of shader stags
*/
enum class ShaderStage {
    COMPUTE = 5,
    RAYGEN = 6,
    INTERSECT = 7,
    ANY_HIT = 8,
    CLOSEST_HIT = 9,
    MISS = 10,
    CALLABLE = 11
};

/**
 * @brief Compiler for GLSL shader code
*/
class HEPHAISTOS_API Compiler {
public:
    /**
     * @brief Add directory to list of paths used to resolve includes
    */
    void addIncludeDir(std::filesystem::path dir);
    /**
     * @brief Removes last added include directory
    */
    void popIncludeDir();
    /**
     * @brief Removes all include directories
    */
    void clearIncludeDir();

    /**
     * @brief Compiles the given GLSL source code
     * 
     * @param code GLSL source code
     * @return Compiled SPIR-V byte code
    */
    [[nodiscard]] std::vector<uint32_t> compile(
        std::string_view code,
        ShaderStage stage = ShaderStage::COMPUTE
    ) const;
    /**
     * @brief Compiled the given GLSL source code
     * 
     * @param code GLSL source code
     * @param headers Map of source code for resolving includes. Takes precedence
     * @return Compiled SPIR-V byte code
    */
    [[nodiscard]] std::vector<uint32_t> compile(
        std::string_view code,
        const HeaderMap& headers,
        ShaderStage stage = ShaderStage::COMPUTE
    ) const;

    Compiler& operator=(Compiler&&) noexcept;
    Compiler(Compiler&&) noexcept;

    Compiler& operator=(const Compiler&) = delete;
    Compiler(const Compiler&) = delete;

    /**
     * @brief Creates a new compiler
    */
    Compiler();
    ~Compiler();

private:
    std::vector<std::filesystem::path> includeDirs;
};

}
