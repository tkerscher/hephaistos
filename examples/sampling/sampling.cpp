#include <iostream>

//shader
#include "sampling.h"

#include <hephaistos/hephaistos.hpp>
using namespace hephaistos;

int main(int argc, char* argv[]) {
    //4 arguments: width, height, input image, output image
    if (argc != 5) {
        std::cerr << "Wrong amount of arguments!\n"
                  << "Usage: sampling width height input output\n";
        return 1;
    }
    auto width = std::stoi(argv[1]);
    auto height = std::stoi(argv[2]);
    auto inPath = argv[3];
    auto outPath = argv[4];

    //create context
    auto context = createContext();

    //load image
    auto texture = ImageBuffer::load(context, inPath).createTexture();
    Image image(context, ImageBuffer::Format, width, height);
    ImageBuffer result(context, width, height);

    //create program
    Program program(context, code);
    program.bindParameterList(texture, image);

    //run program
    beginSequence(context)
        .And(program.dispatch(width, height))
        .Then(retrieveImage(image, result))
        .Submit();

    //write image to disk
    result.save(outPath);
}
