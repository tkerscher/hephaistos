#include <cstdint>
#include <iomanip>
#include <iostream>

//shader
#include "mandelbrot.h"

#include <hephaistos/hephaistos.hpp>
using namespace hephaistos;

//8k resolution
constexpr uint32_t width = 7680;
constexpr uint32_t height = 4320;
//Typed push descriptor
struct Push {
    float transX;
    float transY;
    float scale;
};

int main(int argc, char* argv[]) {
    //Either use default params or parse args
    Push push{ 0.f, 0.f, 1.f };
    if (argc == 4) {
        push.transX = std::atof(argv[1]);
        push.transY = std::atof(argv[2]);
        push.scale = std::atof(argv[3]);
    }

    //create context
    auto context = createContext();

    //print selected device
    auto device = getDeviceInfo(context);
    std::cout << "Selected Device: " << device.name << "\n\n";

    //allocate memory
    ImageBuffer buffer(context, width, height);
    Image image(context, ImageBuffer::Format, width, height);

    //create program
    Program program(context, code);
    program.bindParameter(image, "outImage");

    std::cout << "Rendering...\n";

    //stop the time needed for rendering
    StopWatch watch(context);

    //create sequence
    beginSequence(context)
        .And(watch.start())
        .And(program.dispatch(push, width / 4, height / 4))
        .And(watch.stop())
        .And(retrieveImage(image, buffer)) //has implicit barrier
        .Submit();
    
    //If we don't catch the submission generated by Submit()
    //the execution is paused until it's finished

    //check how long the rendering takes
    auto stamps = watch.getTimeStamps();
    std::cout << std::setprecision(3);
    std::cout << "Rendered in " << (stamps[1] - stamps[0]) * 1e-6 << " ms\n";

    //write image to disk
    std::cout << "Saving...\n";
    buffer.save("mandelbrot.png");
    std::cout << "Done!\n";
}
