#To shorten things
set(INCROOT ${PROJECT_SOURCE_DIR}/include/hephaistos)
set(SRCROOT ${PROJECT_SOURCE_DIR}/src)

#Public API
set(PUBLIC_HEADERS
    ${INCROOT}/buffer.hpp
    ${INCROOT}/command.hpp
    ${INCROOT}/config.hpp
    ${INCROOT}/context.hpp
    ${INCROOT}/handles.hpp
    ${INCROOT}/hephaistos.hpp
    ${INCROOT}/version.hpp
)
#Source
set(SRC
    ${SRCROOT}/buffer.cpp
    ${SRCROOT}/command.cpp
    ${SRCROOT}/context.cpp
    ${SRCROOT}/version.cpp
#vulkan
    ${SRCROOT}/vk/result.hpp
    ${SRCROOT}/vk/types.cpp
    ${SRCROOT}/vk/types.hpp
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
target_compile_features(hephaistos PUBLIC cxx_std_20)

#link libraries
target_link_libraries(hephaistos PRIVATE volk)
target_link_libraries(hephaistos PRIVATE VulkanMemoryAllocator)

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
install(
    TARGETS hephaistos
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hephaistos
)
