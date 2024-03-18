#include <iostream>
#include <sstream>
#include <string>

#include <catch2/catch_session.hpp>

#include "hephaistos/debug.hpp"

using namespace hephaistos;

bool _errorFlag = false; //used by validation error check
void debugCallback(const DebugMessage& message) {
    //Create prefix
    std::string prefix("");
    if (message.severity & DebugMessageSeverityFlagBits::VERBOSE_BIT) prefix = "[VERB]";
    if (message.severity & DebugMessageSeverityFlagBits::INFO_BIT) prefix = "[INFO]";
    if (message.severity & DebugMessageSeverityFlagBits::WARNING_BIT) prefix = "[WARN]";
    if (message.severity & DebugMessageSeverityFlagBits::ERROR_BIT) prefix = "[ERR]";

    //Create message
    std::stringstream stream;
    stream << prefix << "(" << message.idNumber << ": " << message.pIdName << ") " << message.pMessage;

    //Output
    if (message.severity >= DebugMessageSeverityFlagBits::ERROR_BIT) {
        _errorFlag = true; //set flag for testing
        std::cerr << stream.str() << std::endl;
    }
    else {
        std::cout << stream.str() << std::endl;
    }
    fflush(stdout);
}


int main() {
	Catch::Session session;

	//globally enable debug
    if (!isDebugAvailable())
        throw std::runtime_error("Validation Layers are not installed!");
    configureDebug({ .enableAPIValidation = true }, debugCallback);

	//run tests
	return session.run();
}
