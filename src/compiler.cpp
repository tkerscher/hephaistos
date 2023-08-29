#include "hephaistos/compiler.hpp"

#include <memory>
#include <sstream>
#include <stdexcept>

#include <glslang/Include/glslang_c_interface.h>
#include <glslang/Public/resource_limits_c.h>

namespace hephaistos {

namespace {

using Shader = std::unique_ptr<glslang_shader_t, decltype(&glslang_shader_delete)>;
using Program = std::unique_ptr<glslang_program_t, decltype(&glslang_program_delete)>;

glsl_include_result_t* resolve_include(
	void* context,
	const char* header_name,
	const char* includer_name,
	size_t include_depth
) {
	auto headers = static_cast<Compiler::HeaderMap*>(context);
	std::string name(header_name);
	auto pHeader = headers->find(name);
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

int free_include_result(void* context, glsl_include_result_t* result) {
	delete result;
	return 0;
}

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

std::vector<uint32_t> compileImpl(std::string_view code, const Compiler::HeaderMap* headerMap) {
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
	};
	if (headerMap) {
		input.callbacks = {
			.include_system      = resolve_include,
			.include_local       = nullptr,
			.free_include_result = free_include_result
		};
		input.callbacks_ctx = (void*)headerMap;
	}
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

	glslang_program_SPIRV_generate(program, GLSLANG_STAGE_COMPUTE);
	auto size = glslang_program_SPIRV_get_size(program);
	std::vector<uint32_t> result(size);
	glslang_program_SPIRV_get(program, result.data());

	return result;
}

}

std::vector<uint32_t> Compiler::compile(std::string_view code) const {
	return compileImpl(code, nullptr);
}
std::vector<uint32_t> Compiler::compile(std::string_view code, const HeaderMap& headers) const {
	return compileImpl(code, &headers);
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
