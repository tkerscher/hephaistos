cmake_minimum_required(VERSION 3.18...3.22)

include(FetchContent)
include(GNUInstallDirs)

set(FETCHCONTENT_QUIET FALSE)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Make release default target
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

#read version from source
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/src/version.cpp" ver)
string(REGEX MATCH "VERSION_MAJOR = ([0-9]*)" _ ${ver})
set(VERSION_MAJOR ${CMAKE_MATCH_1})
string(REGEX MATCH "VERSION_MINOR = ([0-9]*)" _ ${ver})
set(VERSION_MINOR ${CMAKE_MATCH_1})
string(REGEX MATCH "VERSION_PATCH = ([0-9]*)" _ ${ver})
set(VERSION_PATCH ${CMAKE_MATCH_1})

#create project
project("hephaistos"
    VERSION "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}"
    DESCRIPTION "GPU computing framework using the Vulkan API"
    LANGUAGES C CXX
)
option(BUILD_SHARED_LIBS "Build using shared libraries" OFF)

# fetch volk (vulkan loader)
set(CMAKE_DISABLE_FIND_PACKAGE_Vulkan TRUE)
FetchContent_Declare(
    volk
    GIT_REPOSITORY https://github.com/zeux/volk.git
    GIT_TAG sdk-1.3.239.0
)
FetchContent_MakeAvailable(volk)
set_target_properties(volk PROPERTIES FOLDER "dependencies")
#Fetch vulkan memory allocator
set(VMA_STATIC_VULKAN_FUNCTIONS OFF)
FetchContent_Declare(
    vma
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
    GIT_TAG 66afe099f1cf1f79c270471e9c0f02139072057d
)
FetchContent_GetProperties(vma)
if(NOT vma_POPULATED)
    FetchContent_Populate(vma)
    add_library(VulkanMemoryAllocator INTERFACE)
    target_include_directories(VulkanMemoryAllocator INTERFACE ${vma_SOURCE_DIR}/include)
endif()

#source code
add_subdirectory(src)

#python switch
if(SKBUILD)
    #Python library
    add_subdirectory(python)
else()
    #add stuff we dont want in the python release
    option(BUILD_EXAMPLES "Build examples" ON)
    if(BUILD_EXAMPLES)
        add_subdirectory(examples)
    endif()
endif()
