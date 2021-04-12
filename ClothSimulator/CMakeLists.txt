cmake_minimum_required(VERSION 3.0.0)
project(ClothSim VERSION 0.1.0)


find_package(Vulkan REQUIRED)
if(Vulkan_FOUND)
include_directories(${Vulkan_INCLUDE_DIRS})
endif()

find_package(GLFW3 REQUIRED)
include_directories(./src)
find_package(OpenMP REQUIRED)


add_definitions(-std=c++17)

# ---------------------enable debug symbols---------------------
add_definitions(-g)

# -------------------enable openmp---------------------

if(USE_OPENMP)
    add_definitions(-fopenmp)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_OPENMP")
    message(STATUS "we use openmp")
else()
    message(STATUS "we dont use openmp")
endif()


# ---------------------add address sanitzer---------------------
# set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
# set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")

add_subdirectory(./src/utils)
add_subdirectory(./src/geometries)
add_subdirectory(./src/scenes)
add_subdirectory(./src/shaders)

set(my_libs ${scenes_lib} ${utils_lib})
set(libs ${Vulkan_LIBRARIES} ${GLFW3_LIBRARY} ${my_libs})

add_executable(main ./src/main.cpp)
target_link_libraries(main ${libs} scenes_lib utils_lib geo_lib OpenMP::OpenMP_CXX)

add_custom_command(
    TARGET main POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E rename
        ${CMAKE_CURRENT_BINARY_DIR}/main        
        ${CMAKE_SOURCE_DIR}/main    
)