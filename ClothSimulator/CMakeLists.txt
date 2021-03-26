cmake_minimum_required(VERSION 3.0.0)
project(ClothSim VERSION 0.1.0)

find_package(Vulkan REQUIRED)
if(Vulkan_FOUND)
include_directories(${Vulkan_INCLUDE_DIRS})
endif()

find_package(GLFW3 REQUIRED)
include_directories(./src)
add_definitions(-std=c++17)

add_subdirectory(./src/utils)
add_subdirectory(./src/scenes)
add_subdirectory(./src/shaders)

set(my_libs ${scenes_lib} ${utils_lib})
set(libs ${Vulkan_LIBRARIES} ${GLFW3_LIBRARY} ${my_libs})

add_executable(main ./src/main.cpp)
target_link_libraries(main ${libs} scenes_lib utils_lib)