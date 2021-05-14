pybind11_add_module(depth_sampler ./depth_sampler.cpp ./DepthSampler.cpp)
target_link_libraries(depth_sampler PRIVATE OpenNI2)

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        add_custom_command(
            TARGET depth_sampler POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E rename
                ${CMAKE_CURRENT_BINARY_DIR}/Debug/depth_sampler.lib
                ${CMAKE_SOURCE_DIR}/learning/depth_sampler.lib
        )
        add_custom_command(
            TARGET depth_sampler POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E rename
                ${CMAKE_CURRENT_BINARY_DIR}/Debug/depth_sampler.cp37-win_amd64.pyd
                ${CMAKE_SOURCE_DIR}/learning/depth_sampler.cp37-win_amd64.pyd
        )
else()
    add_custom_command(
        TARGET depth_sampler POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E rename
            ${CMAKE_CURRENT_BINARY_DIR}/Release/depth_sampler.lib
            ${CMAKE_SOURCE_DIR}/learning/depth_sampler.lib
    )
    add_custom_command(
        TARGET depth_sampler POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E rename
            ${CMAKE_CURRENT_BINARY_DIR}/Release/depth_sampler.cp37-win_amd64.pyd
            ${CMAKE_SOURCE_DIR}/learning/depth_sampler.cp37-win_amd64.pyd
    )
endif()