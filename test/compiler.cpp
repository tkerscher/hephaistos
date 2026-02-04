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

TEST_CASE("compiler sessions ensure a common pipeline layout", "[compiler]") {
	auto source1 = R"(
		#version 460
		readonly buffer TensorA { int a[]; };
		readonly buffer TensorC { int c[]; };
		writeonly buffer TensorX { int x[]; };
		void main() {
			uint i = gl_GlobalInvocationID.x;
			x[i] = a[i] + c[i];
		}
	)";
	auto source2 = R"(
		#version 460
		readonly buffer TensorB { int b[]; };
		writeonly buffer TensorY { int y[]; };
		readonly buffer TensorC { int c[]; };
		readonly buffer TensorD { int d[]; };
		void main() {
			uint i = gl_GlobalInvocationID.x;
			y[i] = b[i] + c[i];
		}
	)";

	Compiler compiler;
	CompilerSession session(compiler);
	std::vector<uint32_t> code1, code2;
	REQUIRE_NOTHROW(code1 = session.compile(source1));
	REQUIRE_NOTHROW(code2 = session.compile(source2));
	//we do not expose reflection (yet)
	// -> use the implicit reflection via Program
	std::unique_ptr<Program> program1, program2;
	REQUIRE_NOTHROW(program1 = std::make_unique<Program>(getContext(), code1));
	REQUIRE_NOTHROW(program2 = std::make_unique<Program>(getContext(), code2));

	//check bindings
	REQUIRE(program1->hasBinding("TensorA"));
	REQUIRE(program1->hasBinding("TensorC"));
	REQUIRE(program1->hasBinding("TensorX"));
	REQUIRE(program2->hasBinding("TensorB"));
	REQUIRE(program2->hasBinding("TensorC"));
	REQUIRE(program2->hasBinding("TensorY"));
	REQUIRE(program1->getBindingTraits("TensorC").binding == program2->getBindingTraits("TensorC").binding);
	std::unordered_set<uint32_t> bindings;
	for (auto& b : program1->listBindings())
		bindings.insert(b.binding);
	for (auto& b : program2->listBindings())
		bindings.insert(b.binding);
	REQUIRE(bindings.size() == 5);

	//to be on the safe side, let's check our modifed code actually works
	auto dataB = std::to_array<int32_t>({ 12, 5, 6, 7, 8, 9, 15, 8 });
	auto dataC = std::to_array<int32_t>({ 9, 13, 4, 17, 8, 3, 1, 4 });
	auto dataY = std::to_array<int32_t>({ 21, 18, 10, 24, 16, 12, 16, 12 });
	Tensor<int32_t> tensorB(getContext(), dataB), tensorC(getContext(), dataC), tensorY(getContext(), 8);
	Buffer<int32_t> buffer(getContext(), 8);
	program2->bindParameter(tensorB, "TensorB");
	program2->bindParameter(tensorC, "TensorC");
	program2->bindParameter(tensorY, "TensorY");

	beginSequence(getContext())
		.And(program2->dispatch(8))
		.Then(retrieveTensor(tensorY, buffer))
		.Submit();

	REQUIRE(std::equal(dataY.begin(), dataY.end(), buffer.getMemory().begin()));

	REQUIRE(!hasValidationErrorOccurred());
}
