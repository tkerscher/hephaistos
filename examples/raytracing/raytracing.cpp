#include <array>
#include <iostream>

#include <hephaistos/hephaistos.hpp>
#include <hephaistos/raytracing.hpp>

using namespace hephaistos;

int main() {
	//create context with ray tracing enabled
	auto context = createEmptyContext();
	try {
		auto extensions = std::to_array({
			createRaytracingExtension()
		});
		context = createContext(extensions);
		return 1;
	}
	catch (const std::exception& e) {
		std::cerr << "Failed to create context!\n"
				  << e.what() << '\n';
	}

	//print selected device
	auto device = getDeviceInfo(context);
	std::cout << "Selected Device: " << device.name << "\n\n";
}
