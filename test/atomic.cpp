#include <catch2/catch_test_macros.hpp>

#include <algorithm>

#include <hephaistos/atomic.hpp>

using namespace hephaistos;

//It's hard to test the atomic extensions, so we just see
//whether we can enable all available ones

TEST_CASE("atomic extension can be used", "[atomic]") {
	//query properties of first device
	auto devices = enumerateDevices();
	auto& device = devices.front();
	auto props = getAtomicsProperties(device);
	auto ext = createAtomicsExtension(props);
	auto context = hephaistos::createContext(device, { &ext, 1 });

	//check enabled props are the same
	auto& enabled = getEnabledAtomics(context);
	bool same = //trying to be smart here is most likely UB
		props.bufferInt64Atomics == enabled.bufferInt64Atomics &&
		props.bufferFloat16Atomics == enabled.bufferFloat16Atomics &&
		props.bufferFloat16AtomicAdd == enabled.bufferFloat16AtomicAdd &&
		props.bufferFloat16AtomicMinMax == enabled.bufferFloat16AtomicMinMax &&
		props.bufferFloat32Atomics == enabled.bufferFloat32Atomics &&
		props.bufferFloat32AtomicAdd == enabled.bufferFloat32AtomicAdd &&
		props.bufferFloat32AtomicMinMax == enabled.bufferFloat32AtomicMinMax &&
		props.bufferFloat64Atomics == enabled.bufferFloat64Atomics &&
		props.bufferFloat64AtomicAdd == enabled.bufferFloat64AtomicAdd &&
		props.bufferFloat64AtomicMinMax == enabled.bufferFloat64AtomicMinMax &&
		props.sharedInt64Atomics == enabled.sharedInt64Atomics &&
		props.sharedFloat16Atomics == enabled.sharedFloat16Atomics &&
		props.sharedFloat16AtomicAdd == enabled.sharedFloat16AtomicAdd &&
		props.sharedFloat16AtomicMinMax == enabled.sharedFloat16AtomicMinMax &&
		props.sharedFloat32Atomics == enabled.sharedFloat32Atomics &&
		props.sharedFloat32AtomicAdd == enabled.sharedFloat32AtomicAdd &&
		props.sharedFloat32AtomicMinMax == enabled.sharedFloat32AtomicMinMax &&
		props.sharedFloat64Atomics == enabled.sharedFloat64Atomics &&
		props.sharedFloat64AtomicAdd == enabled.sharedFloat64AtomicAdd &&
		props.sharedFloat64AtomicMinMax == enabled.sharedFloat64AtomicMinMax &&
		props.imageInt64Atomics == enabled.imageInt64Atomics &&
		props.imageFloat32Atomics == enabled.imageFloat32Atomics &&
		props.imageFloat32AtomicAdd == enabled.imageFloat32AtomicAdd &&
		props.imageFloat32AtomicMinMax == enabled.imageFloat32AtomicMinMax;
	REQUIRE(same);
}
