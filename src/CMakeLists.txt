#To shorten things
set(INCROOT ${PROJECT_SOURCE_DIR}/include/hephaistos)
set(SRCROOT ${PROJECT_SOURCE_DIR}/src)

#Public API
set(PUBLIC_HEADERS
    ${INCROOT}/argument.hpp
    ${INCROOT}/atomic.hpp
    ${INCROOT}/buffer.hpp
    ${INCROOT}/command.hpp
    ${INCROOT}/compiler.hpp
    ${INCROOT}/config.hpp
    ${INCROOT}/context.hpp
    ${INCROOT}/debug.hpp
    ${INCROOT}/handles.hpp
    ${INCROOT}/hephaistos.hpp
    ${INCROOT}/image.hpp
    ${INCROOT}/imageformat.hpp
    ${INCROOT}/program.hpp
    ${INCROOT}/raytracing.hpp
    ${INCROOT}/stopwatch.hpp
    ${INCROOT}/types.hpp
    ${INCROOT}/version.hpp
)
#Source
set(SRC
    ${SRCROOT}/atomic.cpp
    ${SRCROOT}/buffer.cpp
    ${SRCROOT}/command.cpp
    ${SRCROOT}/compiler.cpp
    ${SRCROOT}/context.cpp
    ${SRCROOT}/debug.cpp     
    ${SRCROOT}/image.cpp
    ${SRCROOT}/program.cpp
    ${SRCROOT}/raytracing.cpp
    ${SRCROOT}/stopwatch.cpp
    ${SRCROOT}/types.cpp
    ${SRCROOT}/version.cpp
#vulkan
    ${SRCROOT}/vk/result.hpp
    ${SRCROOT}/vk/instance.cpp
    ${SRCROOT}/vk/instance.hpp
    ${SRCROOT}/vk/types.cpp
    ${SRCROOT}/vk/types.hpp
    ${SRCROOT}/vk/util.hpp
    ${SRCROOT}/vk/vma.cpp
)

#create library
add_library(hephaistos)
target_sources(hephaistos PRIVATE ${SRC})
target_include_directories(hephaistos
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PRIVATE ${PROJECT_SOURCE_DIR}/src
)
set_target_properties(hephaistos PROPERTIES PUBLIC_HEADER "${PUBLIC_HEADERS}")
set_target_properties(hephaistos PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(hephaistos PUBLIC cxx_std_20)

#link libraries
target_link_libraries(hephaistos PRIVATE volk)
target_link_libraries(hephaistos PRIVATE VulkanMemoryAllocator)
target_link_libraries(hephaistos PRIVATE spirv-reflect)
target_link_libraries(hephaistos PRIVATE stb)
target_link_libraries(hephaistos PRIVATE glslang)

#tell the compiler to hide all symbols by default
set_target_properties(hephaistos PROPERTIES CXX_VISIBILITY hidden)
set_target_properties(hephaistos PROPERTIES CMAKE_VISIBILITY_INLINES_HIDDEN 1)
target_compile_definitions(hephaistos PRIVATE HEPHAISTOS_EXPORTS)
#We need an extra switch for static builds
if (NOT BUILD_SHARED_LIBS)
    target_compile_definitions(hephaistos PUBLIC HEPHAISTOS_STATIC)
endif()

#postfix for static and/or debugs
if(BUILD_SHARED_LIBS)
    set_target_properties(hephaistos PROPERTIES DEBUG_POSTFIX -d)
else()
    set_target_properties(hephaistos PROPERTIES DEBUG_POSTFIX -s-d)
    set_target_properties(hephaistos PROPERTIES RELEASE_POSTFIX -s)
    set_target_properties(hephaistos PROPERTIES MINSIZEREL_POSTFIX -s)
    set_target_properties(hephaistos PROPERTIES RELWITHDEBINFO_POSTFIX -s)
endif()

#install lib
if(NOT SKBUILD)
    install(
        TARGETS hephaistos
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hephaistos
    )
endif()
