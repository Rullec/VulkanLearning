pybind11_add_module(video_manager ./video_manager_pybind_export.cpp ./VideoManager.cpp)
target_link_libraries(video_manager PRIVATE OpenNI2)

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        # add_custom_command(
        #     TARGET video_manager POST_BUILD
        #     COMMAND ${CMAKE_COMMAND} -E rename
        #         ${CMAKE_CURRENT_BINARY_DIR}/Debug/video_manager.lib
        #         ${CMAKE_SOURCE_DIR}/learning/video_manager.lib
        # )
        add_custom_command(
            TARGET video_manager POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E rename
                ${CMAKE_CURRENT_BINARY_DIR}/Debug/video_manager.cp37-win_amd64.pyd
                ${CMAKE_SOURCE_DIR}/learning/video_manager.cp37-win_amd64.pyd
        )
else()
    # add_custom_command(
    #     TARGET video_manager POST_BUILD
    #     COMMAND ${CMAKE_COMMAND} -E rename
    #         ${CMAKE_CURRENT_BINARY_DIR}/Release/video_manager.lib
    #         ${CMAKE_SOURCE_DIR}/learning/video_manager.lib
    # )
    add_custom_command(
        TARGET video_manager POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E rename
            ${CMAKE_CURRENT_BINARY_DIR}/Release/video_manager.cp37-win_amd64.pyd
            ${CMAKE_SOURCE_DIR}/learning/video_manager.cp37-win_amd64.pyd
    )
endif()