#include <iostream>

#include "hephaistos/hephaistos.hpp"
namespace hp = hephaistos;

int main() {
    //print header
    std::cout << "Hephaistos v" << hp::VERSION_MAJOR << "." << hp::VERSION_MINOR << "." << hp::VERSION_PATCH << std::endl;

    //fetch devices
    auto devices = hp::enumerateDevices();
    std::cout << devices.size() << " device(s) found.\n";

    std::cout << std::boolalpha;
    //iterate over all devices;
    for (auto& device : devices) {
        auto info = hp::getDeviceInfo(device);
        std::cout << "\n";
        std::cout << "Name:     " << info.name << '\n';
        std::cout << "Discrete: " << info.isDiscrete << '\n';
    }

    //done
    return 0;
}
