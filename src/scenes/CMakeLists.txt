add_library(scenes_lib 
DrawScene.cpp 
DrawSceneVk.cpp 
SceneBuilder.cpp
Scene.cpp
SimScene.cpp
SemiImplicitScene.cpp
BaraffScene.cpp
ImplicitScene.cpp
LinctexScene.cpp
SynDataScene.cpp
PDScene.cpp
ProcessTrainDataScene.cpp
PBDScene.cpp
Perturb.cpp
cameras/CameraBase.cpp
cameras/FPSCamera.cpp
cameras/ArcBallCamera.cpp
cameras/CameraUtil.cpp
)
target_link_libraries(scenes_lib ${GLFW3_LIBRARY} ${Vulkan_LIBRARIES} sim_lib)

if(WIN32)
if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    # message(FATAL_ERROR debug)
    target_link_libraries(scenes_lib Simulationd)
else()
    # message(FATAL_ERROR release)
    target_link_libraries(scenes_lib Simulation)
endif()
endif()