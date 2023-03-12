#include <iostream>

#include <hephaistos/hephaistos.hpp>

using namespace hephaistos;

int main() {
	//create context
	auto context = createContext();

	//print selected device
	auto device = getDeviceInfo(context);
	std::cout << "Selected Device: " << device.name << "\n\n";

	//create memory
	Buffer buffer1(context, { 2u, 4u, 8u, 16u, 32u, 64u, 128u, 256u, 512u, 1024u });
	Buffer<uint32_t> buffer2(context, 10);
	Tensor<uint32_t> tensor(context, 10);

	//copy data
	Timeline timeline(context);
	auto submission = beginSequence(timeline)
		.And(updateTensor(buffer1, tensor))
		.Then(retrieveTensor(tensor, buffer2))
		.Submit();

	//wait for roundtrip
	std::cout << "Uploading Data...\n";
	timeline.waitValue(1);
	std::cout << "Fetching Data..\n";
	timeline.waitValue(2);

	//read out data
	for (auto& v : buffer2.getMemory()) {
		std::cout << v << '\n';
	}
}
