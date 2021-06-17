add_library(draw_scene_lib
DrawScene.cpp 
DrawSceneVk.cpp 
DrawSceneShot.cpp 
)

add_library(sim_scenes_lib 
Scene.cpp
SimScene.cpp
Perturb.cpp
LinctexScene.cpp
LinctexCloth.cpp
EmptyCloth.cpp
MeshVisScene.cpp
SynDataScene.cpp
ProcessTrainDataScene.cpp
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
        target_link_libraries(scenes_lib Simulationd)
    else()
        target_link_libraries(scenes_lib Simulation)
    endif()
endif()

if(WIN32)
    pybind11_add_module(process_data_scene ./process_traindata_scene_pybind_export.cpp ./ProcessTrainDataScene.cpp)

    if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        target_link_libraries(process_data_scene PRIVATE sim_scenes_lib scenes_lib Simulationd)
    else()
        target_link_libraries(process_data_scene PRIVATE sim_scenes_lib scenes_lib Simulation)
    endif()

    


    move_pybind11_release_to_dir(${DLL_OUTPUT_DIR} process_data_scene)

endif()