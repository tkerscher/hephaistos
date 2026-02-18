#include <catch2/catch_test_macros.hpp>

#include <array>

#include <hephaistos/context.hpp>
#include <hephaistos/command.hpp>
#include <hephaistos/debug.hpp>
#include <hephaistos/program.hpp>

#include "validation.hpp"

//shader code
#include "shader/deviceloss.h"

using namespace hephaistos;

namespace {

auto extensions = std::to_array({
    createDeviceFaultInfoExtension()
});

}

//The following test is successful on its own, but causes the following
//ones to fail. Unfortunately, crashing the GPU on purpose seems to
//cause bigger issues in the driver. While the spec allows to create
//new logical devices from an existing instance, the driver always
//returns VK_ERROR_INITIALIZATION_FAILED after a VK_ERROR_DEVICE_LOST.
//For now, we will not include this test.
//TEST_CASE("device fault gives info about device loss", "[debug]") {
//    if (!isDeviceSuitable(extensions)) {
//        SKIP("Device fault extension is not supported");
//    }
//
//    //provoke device loss by writing to null pointer on GPU
//    auto context = createContext(extensions);
//    Program program(context, deviceloss_code);
//    beginSequence(context).And(program.dispatch(32)).Submit();
//
//    //check we succssfully killed the GPU
//    REQUIRE_THROWS(checkDeviceHealth(context));
//
//    //we cannot check the content, but we can check whether the func runs through
//    DeviceFaultInfo info;
//    REQUIRE_NOTHROW(info = getDeviceFaultInfo(context));
//
//    //clear any validation errors
//    hasValidationErrorOccurred();
//}
