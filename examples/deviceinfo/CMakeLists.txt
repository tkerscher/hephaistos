add_executable(deviceinfo deviceinfo.cpp)
target_link_libraries(deviceinfo hephaistos)
set_target_properties(deviceinfo PROPERTIES FOLDER "examples")

install(
	TARGETS deviceinfo
	RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
	COMPONENT examples
)
