#include <array>
#include <cstdint>
#include <iostream>

#include <hephaistos/hephaistos.hpp>
#include <hephaistos/raytracing.hpp>

struct Vertex{
    float position[3];
    float normal[3];
};

//#include "model_prism.h"
#include "model_sphere.h"
//#include "model_monkey.h"
#include "raytracing.h"

//typed push descriptor
struct Push {
	uint64_t vertices_address;
	uint64_t indices_address;
};

using namespace hephaistos;

int main() {
	//create context with ray tracing enabled
	auto context = createEmptyContext();
	try {
		auto extensions = std::to_array({
			createRaytracingExtension()
		});
		context = createContext(extensions);
	}
	catch (const std::exception& e) {
		std::cerr << "Failed to create context!\n"
				  << e.what() << '\n';
		return 1;
	}

	//print selected device
	auto device = getDeviceInfo(context);
	std::cout << "Selected Device: " << device.name << "\n\n";

	//create acceleration structure
	auto vertexSpan = std::span<const Vertex>(vertices);
	Mesh mesh{
		.vertices = std::as_bytes(std::span<const Vertex>(vertices)),
		.vertexStride = sizeof(Vertex),
		.indices = indices
	};
	GeometryStore geometries(context, mesh, true);
	auto instance = geometries.createInstance(0);
	AccelerationStructure accStruct(context, instance);

	//Fetch geometry data pointers and store in push descriptor
	Push push{
		.vertices_address = geometries[0].vertices_address,
		.indices_address = geometries[0].indices_address
	};
	//create output image and buffer
	ImageBuffer imgBuffer(context, 1024, 1024);
	auto image = imgBuffer.createImage(false);

	//create program
	Program program(context, code);
	program.bindParameterList(accStruct, image);

	//Create sequence
	beginSequence(context)
		.And(program.dispatch(push, 256, 256))
		.Then(retrieveImage(image, imgBuffer))
		.Submit();

	//write image to disk
	imgBuffer.save("raytracing.png");
}
