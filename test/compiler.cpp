#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <fstream>

#include <hephaistos/buffer.hpp>
#include <hephaistos/command.hpp>
#include <hephaistos/context.hpp>
#include <hephaistos/compiler.hpp>
#include <hephaistos/program.hpp>

#include "validation.hpp"

using namespace hephaistos;

namespace {

ContextHandle getContext() {
	static ContextHandle context = createEmptyContext();
	if (!context)
		context = createContext();
	return context;
}

std::array<int32_t, 4> dataA{ { 12, 4156, 12, 56 } };
std::array<int32_t, 4> dataB{ { 17, 12, 123, 3 } };
std::array<int32_t, 4> dataOut{ { 29, 4168, 135, 59 } };
std::array<int32_t, 4> dataOut2{ { 41, 8324, 147, 115 } };

}

TEST_CASE("compiler can compile GLSL code to SPIR-V", "[compiler]") {
	auto source = R"(
		#version 460

		layout(local_size_x = 1) in;

		readonly buffer tensorA { int in_a[]; };
		readonly buffer tensorB { int in_b[]; };
		writeonly buffer tensorOut { int out_c[]; };

		void main() {
			uint idx = gl_GlobalInvocationID.x;
			out_c[idx] = in_a[idx] + in_b[idx];
		}
	)";

	Compiler compiler;
	std::vector<uint32_t> code;
	REQUIRE_NOTHROW(code = compiler.compile(source));
	REQUIRE(code.size() > 0);

	std::unique_ptr<Program> program;
	REQUIRE_NOTHROW(program = std::make_unique<Program>(getContext(), code));

	Tensor<int32_t> tensorA(getContext(), dataA);
	Tensor<int32_t> tensorB(getContext(), dataB);
	Tensor<int32_t> tensorOut(getContext(), 4);
	Buffer<int32_t> buffer(getContext(), 4);

	program->bindParameterList(tensorA, tensorB, tensorOut);

	beginSequence(getContext())
		.And(program->dispatch(4))
		.Then(retrieveTensor(tensorOut, buffer))
		.Submit();

	REQUIRE(std::equal(dataOut.begin(), dataOut.end(), buffer.getMemory().begin()));

	REQUIRE(!hasValidationErrorOccurred());
}

TEST_CASE("Compiler can compile stages from ray tracing pipeline", "[compiler]") {
	auto source = R"(
		#version 460
		#extension GL_EXT_ray_tracing : enable

		layout(binding = 0) uniform accelerationStructureEXT tlas;
		layout(binding = 1, rgba8) uniform image2D outImage;

		layout(location = 0) rayPayloadEXT vec3 hitValue;

		void main() {
			vec2 size = vec2(imageSize(outImage));
			vec2 pos = vec2(gl_LaunchIDEXT.xy);
			vec2 coord = pos / size * 2 - vec2(1.0, 1.0);

			//to things a bit more simple, we render orthogonal in y direction
			//image is in the xz plane; flip y as usual
			vec3 start = vec3(coord.x, -2.0, -coord.y);
			vec3 dir = vec3(0.0, 1.0, 0.0);

			traceRayEXT(
				tlas,
				gl_RayFlagsOpaqueEXT, 0xFF,
				0, 0, 0,
				start, 0.0,
				dir, 4.0,
				0
			);

			imageStore(outImage, ivec2(gl_LaunchIDEXT.xy), vec4(hitValue, 1.0));
		}
	)";

	Compiler compiler;
	std::vector<uint32_t> code;
	REQUIRE_NOTHROW(code = compiler.compile(source, ShaderStage::RAYGEN));
	REQUIRE(code.size() > 0);
}

TEST_CASE("compiler can include files from header map", "[compiler]") {
	HeaderMap headers = {
		{ "foo.glsl", "int foo(int a, int b) {\n\treturn 2 * a + b;\n}\n" },
		{ "bar.glsl", "#include \"foo.glsl\"\n\nint bar(int a, int b) {\n\treturn foo(a,b);\n}\n" }
	};

	auto source = R"(
		#version 460
		#extension GL_GOOGLE_include_directive: require
		#include "bar.glsl"

		layout(local_size_x = 1) in;

		readonly buffer tensorA { int in_a[]; };
		readonly buffer tensorB { int in_b[]; };
		writeonly buffer tensorOut { int out_c[]; };

		void main() {
			uint idx = gl_GlobalInvocationID.x;
			out_c[idx] = bar(in_a[idx], in_b[idx]);
		}
	)";

	Compiler compiler;
	std::vector<uint32_t> code;
	REQUIRE_NOTHROW(code = compiler.compile(source, headers));
	REQUIRE(code.size() > 0);

	std::unique_ptr<Program> program;
	REQUIRE_NOTHROW(program = std::make_unique<Program>(getContext(), code));

	Tensor<int32_t> tensorA(getContext(), dataA);
	Tensor<int32_t> tensorB(getContext(), dataB);
	Tensor<int32_t> tensorOut(getContext(), 4);
	Buffer<int32_t> buffer(getContext(), 4);

	program->bindParameterList(tensorA, tensorB, tensorOut);

	beginSequence(getContext())
		.And(program->dispatch(4))
		.Then(retrieveTensor(tensorOut, buffer))
		.Submit();

	REQUIRE(std::equal(dataOut2.begin(), dataOut2.end(), buffer.getMemory().begin()));

	REQUIRE(!hasValidationErrorOccurred());
}

TEST_CASE("compiler can fetch headers from disk", "[compiler]") {
	//create temp directory
	//hard coded path is bad, but what are the chances of collision, right? Right?
	auto tmp_dir = std::filesystem::temp_directory_path() / "hephaistos_test_EF45A4C2";
	std::filesystem::create_directory(tmp_dir);
	//nice trick to mimick defer in golang or zig
	std::shared_ptr<void> defer_tmp_del(nullptr, [&tmp_dir](...){
		std::filesystem::remove_all(tmp_dir);
	});

	//create two source files
	{
		std::ofstream file(tmp_dir / "foo.glsl");
		file << "int foo(int a, int b) {\n\treturn 2 * a + b;\n}\n";
		file.close();
	}
	{
		std::ofstream file(tmp_dir / "bar.glsl");
		file << "#include \"foo.glsl\"\n\nint bar(int a, int b) {\n\treturn foo(a,b);\n}\n";
		file.close();
	}

	auto source = R"(
		#version 460
		#extension GL_GOOGLE_include_directive: require
		#include "bar.glsl"

		layout(local_size_x = 1) in;

		readonly buffer tensorA { int in_a[]; };
		readonly buffer tensorB { int in_b[]; };
		writeonly buffer tensorOut { int out_c[]; };

		void main() {
			uint idx = gl_GlobalInvocationID.x;
			out_c[idx] = bar(in_a[idx], in_b[idx]);
		}
	)";

	Compiler compiler;
	compiler.addIncludeDir(tmp_dir);
	std::vector<uint32_t> code;
	REQUIRE_NOTHROW(code = compiler.compile(source));
	REQUIRE(code.size() > 0);

	std::unique_ptr<Program> program;
	REQUIRE_NOTHROW(program = std::make_unique<Program>(getContext(), code));

	Tensor<int32_t> tensorA(getContext(), dataA);
	Tensor<int32_t> tensorB(getContext(), dataB);
	Tensor<int32_t> tensorOut(getContext(), 4);
	Buffer<int32_t> buffer(getContext(), 4);

	program->bindParameterList(tensorA, tensorB, tensorOut);

	beginSequence(getContext())
		.And(program->dispatch(4))
		.Then(retrieveTensor(tensorOut, buffer))
		.Submit();

	REQUIRE(std::equal(dataOut2.begin(), dataOut2.end(), buffer.getMemory().begin()));

	REQUIRE(!hasValidationErrorOccurred());
}
