cmake_minimum_required(VERSION 3.0.0)
project(ClothSim VERSION 0.1.0)


find_package(Vulkan REQUIRED)
if(Vulkan_FOUND)
include_directories(${Vulkan_INCLUDE_DIRS})
endif()

find_package(GLFW3 REQUIRED)
include_directories(./src)

# add address sanitzer
# add_definitions(-std=c++17 -fsanitize=address -fno-omit-frame-pointer)
add_definitions(-std=c++17)
# set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
# set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")

add_subdirectory(./src/utils)
add_subdirectory(./src/scenes)
add_subdirectory(./src/shaders)

set(my_libs ${scenes_lib} ${utils_lib})
set(libs ${Vulkan_LIBRARIES} ${GLFW3_LIBRARY} ${my_libs})

add_executable(main ./src/main.cpp)
target_link_libraries(main ${libs} scenes_lib utils_lib)