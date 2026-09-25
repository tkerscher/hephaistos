#include <catch2/catch_test_macros.hpp>

#include <hephaistos/context.hpp>
#include <hephaistos/types.hpp>

using namespace hephaistos;

TEST_CASE("can make type support requirement via extension", "[types]") {
	//we don't know on which hardware we run this
	//-> require the types the hardware reports as supported
	auto devices = enumerateDevices();
	REQUIRE(devices.size() >= 1);
	const auto supported = getSupportedTypes(devices[0]);
	auto ext = createTypeExtension(supported);
	auto context = createContext(devices[0], { &ext, 1 });
	const auto reported = getSupportedTypes(context);

	REQUIRE(supported.float16 == reported.float16);
	REQUIRE(supported.float64 == reported.float64);
	REQUIRE(supported.int64 == reported.int64);
	REQUIRE(supported.int16 == reported.int16);
	REQUIRE(supported.int8 == reported.int8);
}

TEST_CASE("can make FMA support via extension", "[types]") {
	//we don't know which hardware we run this on
	//-> require the support the hardware reports
	auto devices = enumerateDevices();
	REQUIRE(devices.size() >= 1);
	const auto supported = getFmaSupport(devices[0]);
	auto ext = createFmaExtension(supported);
	auto context = createContext(devices[0], { &ext, 1 });
	const auto reported = getFmaSupport(context);

	REQUIRE(supported.float16 == reported.float16);
	REQUIRE(supported.float32 == reported.float32);
	REQUIRE(supported.float64 == reported.float64);
}
