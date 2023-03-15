add_executable(raytracing raytracing.cpp)
target_link_libraries(raytracing hephaistos)
set_target_properties(raytracing PROPERTIES FOLDER "examples")

install(
	TARGETS raytracing
	RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
	COMPONENT examples
)
