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

	//load data to device
	Tensor<Vertex> vertexTensor(context, vertices);
	Tensor<uint32_t> indexTensor(context, indices);

	//create acceleration structure
	auto vertexSpan = std::span<const Vertex>(vertices);
	Geometry geometry{
		.vertices = std::as_bytes(std::span<const Vertex>(vertices)),
		.vertexStride = sizeof(Vertex),
		.vertexCount = static_cast<uint32_t>(vertexSpan.size()),
		.indices = indices
	};
	GeometryInstance geoInstance{
		.geometry = std::cref(geometry)
	};
	AccelerationStructure accStruct(context, geoInstance);

	//create output image and buffer
	ImageBuffer imgBuffer(context, 1024, 1024);
	auto image = imgBuffer.createImage(false);

	//create program
	Program program(context, code);
	program.bindParameterList(accStruct, vertexTensor, indexTensor, image);

	//Create sequence
	Timeline timeline(context);
	beginSequence(timeline)
		.And(program.dispatch(256, 256))
		.Then(retrieveImage(image, imgBuffer))
		.Submit();

	//write image to disk
	imgBuffer.save("raytracing.png");
}
