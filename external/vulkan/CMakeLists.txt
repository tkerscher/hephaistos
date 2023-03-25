add_library(Vulkan-Headers INTERFACE)
target_include_directories(Vulkan-Headers INTERFACE ${CMAKE_CURRENT_LIST_DIR})
set_target_properties(Vulkan-Headers PROPERTIES FOLDER "dependencies")
