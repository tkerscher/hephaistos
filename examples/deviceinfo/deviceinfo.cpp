#include <iostream>

#include "hephaistos/hephaistos.hpp"
namespace hp = hephaistos;

int main() {
    //print header
    std::cout << "Hephaistos v" << hp::VERSION_MAJOR << "." << hp::VERSION_MINOR << "." << hp::VERSION_PATCH << std::endl;

    //done
    return 0;
}
