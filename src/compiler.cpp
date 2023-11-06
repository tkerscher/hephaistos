#include "hephaistos/compiler.hpp"

#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string.h>

#include <glslang/Include/glslang_c_interface.h>
#include <glslang/Public/resource_limits_c.h>

namespace hephaistos {

namespace {

using CompilerContext = std::pair<const Compiler::HeaderMap*, const std::vector<std::filesystem::path>* >;

using Shader = std::unique_ptr<glslang_shader_t, decltype(&glslang_shader_delete)>;
using Program = std::unique_ptr<glslang_program_t, decltype(&glslang_program_delete)>;

glsl_include_result_t* resolve_include_map(
    void* context,
    const char* header_name,
    const char* includer_name,
    size_t include_depth
) {
    auto headers = static_cast<CompilerContext*>(context)->first;
    if (!headers)
        return nullptr;
    auto pHeader = headers->find(header_name);
    if (pHeader != headers->end()) {
        auto result = new glsl_include_result_t;
        result->header_name = pHeader->first.c_str();
        result->header_data = pHeader->second.c_str();
        result->header_length = pHeader->second.size();
        return result;
    }
    else {
        return nullptr;
    }
}

glsl_include_result_t* resolve_include_dirs(
    void* context,
    const char* header_name,
    const char* includer_name,
    size_t include_depth
) {
    auto& dirs = *static_cast<CompilerContext*>(context)->second;
    //check all directories
    for (auto& dir : dirs) {
        auto path = dir / header_name;
        if (std::filesystem::exists(path) && std::filesystem::is_regular_file(path)) {
            auto result = new glsl_include_result_t;
            size_t size = std::filesystem::file_size(path);
            auto data = new char[size];
            //load data
            std::ifstream file(path, std::ios::binary);
            if (file.read(data, size)) {
                result->header_name = strdup(header_name);
                result->header_length = size;
                result->header_data = data;
                return result;
            }
            else {
                //error -> clean up
                delete[] data;
                return result;
            }
        }
    }

    //nothing found
    return nullptr;
}

int free_include_result(void* context, glsl_include_result_t* result) {
    //check if local or loaded from disk
    auto headers = static_cast<CompilerContext*>(context)->first;
    if (!headers || !headers->contains(result->header_name)) {
        //from disk -> free resources allocated in result
        delete[] result->header_name;
        delete[] result->header_data;
    }
    delete result;
    return 0;
}

const glsl_include_callbacks_t includeCallbacks{
    .include_system = resolve_include_dirs,
    .include_local = resolve_include_map,
    .free_include_result = free_include_result
};

void compile_error(const char* reason, glslang_shader_t* shader) {
    std::stringstream sstream;
    sstream << reason << '\n' << glslang_shader_get_info_log(shader) << '\n' << glslang_shader_get_info_debug_log(shader);
    throw std::runtime_error(sstream.str());
}

void compile_error(const char* reason, glslang_program_t* program) {
    std::stringstream sstream;
    sstream << reason << '\n' << glslang_program_get_info_log(program) << '\n' << glslang_program_get_info_debug_log(program);
    throw std::runtime_error(sstream.str());
}

std::vector<uint32_t> compileImpl(std::string_view code, void* callbacks_ctx = nullptr) {
    glslang_input_t input = {
        .language = GLSLANG_SOURCE_GLSL,
        .stage = GLSLANG_STAGE_COMPUTE,
        .client = GLSLANG_CLIENT_VULKAN,
        .client_version = GLSLANG_TARGET_VULKAN_1_2,
        .target_language = GLSLANG_TARGET_SPV,
        .target_language_version = GLSLANG_TARGET_SPV_1_4,
        .code = code.data(),
        .default_version = 460,
        .default_profile = GLSLANG_NO_PROFILE,
        .force_default_version_and_profile = false,
        .forward_compatible = false,
        .messages = GLSLANG_MSG_DEFAULT_BIT,
        .resource = glslang_default_resource(),
        .callbacks = includeCallbacks,
        .callbacks_ctx = callbacks_ctx
    };
    Shader shaderHandle{ glslang_shader_create(&input), glslang_shader_delete };
    auto shader = shaderHandle.get();

    glslang_shader_set_options(shader, GLSLANG_SHADER_AUTO_MAP_BINDINGS);

    if (!glslang_shader_preprocess(shader, &input))
        compile_error("GLSL Preprocessing failed!", shader);
    if (!glslang_shader_parse(shader, &input))
        compile_error("GLSL Parsing failed!", shader);

    Program programHandle{ glslang_program_create(), glslang_program_delete };
    auto program = programHandle.get();
    glslang_program_add_shader(program, shader);

    if (!glslang_program_link(program, GLSLANG_MSG_SPV_RULES_BIT | GLSLANG_MSG_VULKAN_RULES_BIT))
        compile_error("GLSL Linking failed!", program);

    if (!glslang_program_map_io(program))
        compile_error("GLSL binding remapping failed!", program);

    glslang_spv_options_t spv_options{
        .optimize_size = true
    };
    glslang_program_SPIRV_generate_with_options(program, GLSLANG_STAGE_COMPUTE, &spv_options);
    auto size = glslang_program_SPIRV_get_size(program);
    std::vector<uint32_t> result(size);
    glslang_program_SPIRV_get(program, result.data());

    return result;
}

}

std::vector<uint32_t> Compiler::compile(std::string_view code) const {
    CompilerContext context(nullptr, &includeDirs);
    return compileImpl(code, &context);
}
std::vector<uint32_t> Compiler::compile(std::string_view code, const HeaderMap& headers) const {
    CompilerContext context(&headers, &includeDirs);
    return compileImpl(code, &context);
}

void Compiler::addIncludeDir(std::filesystem::path path) {
    includeDirs.emplace_back(std::move(path));
}
void Compiler::popIncludeDir() {
    includeDirs.pop_back();
}
void Compiler::clearIncludeDir() {
    includeDirs.clear();
}

Compiler& Compiler::operator=(Compiler&&) noexcept = default;
Compiler::Compiler(Compiler&&) noexcept = default;

Compiler::Compiler()
{
    glslang_initialize_process();
}

Compiler::~Compiler() {
    glslang_finalize_process();
}

}
