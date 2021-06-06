add_library(draw_scene_lib
DrawScene.cpp 
DrawSceneVk.cpp 
DrawSceneShot.cpp 
)

add_library(sim_scenes_lib 
Scene.cpp
SimScene.cpp
SemiImplicitScene.cpp
BaraffScene.cpp
ImplicitScene.cpp
LinctexScene.cpp
SynDataScene.cpp
MeshVisScene.cpp
PDScene.cpp
ProcessTrainDataScene.cpp
PBDScene.cpp
Perturb.cpp
cameras/CameraBase.cpp
cameras/FPSCamera.cpp
cameras/ArcBallCamera.cpp
cameras/CameraUtil.cpp
)

add_library(
    scenes_lib
    SceneBuilder.cpp
)

target_link_libraries(sim_scenes_lib sim_lib geo_lib utils_lib)

target_link_libraries(scenes_lib ${GLFW3_LIBRARY} ${Vulkan_LIBRARIES}  draw_scene_lib sim_scenes_lib)

if(WIN32)
if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    # message(FATAL_ERROR debug)
    target_link_libraries(scenes_lib Simulationd)
else()
    # message(FATAL_ERROR release)
    target_link_libraries(scenes_lib Simulation)
endif()
endif()

if(WIN32)
    pybind11_add_module(process_data_scene ./process_traindata_scene_pybind_export.cpp ./ProcessTrainDataScene.cpp)
    target_link_libraries(process_data_scene PRIVATE sim_scenes_lib geo_lib)

    if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
            add_custom_command(
                TARGET process_data_scene POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E rename
                    ${CMAKE_CURRENT_BINARY_DIR}/Debug/process_data_scene.cp37-win_amd64.pyd
                    ${CMAKE_SOURCE_DIR}/learning/process_data_scene.cp37-win_amd64.pyd
            )
    else()
        add_custom_command(
            TARGET process_data_scene POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E rename
                ${CMAKE_CURRENT_BINARY_DIR}/Release/process_data_scene.cp37-win_amd64.pyd
                ${CMAKE_SOURCE_DIR}/learning/process_data_scene.cp37-win_amd64.pyd
        )
    endif()

endif()