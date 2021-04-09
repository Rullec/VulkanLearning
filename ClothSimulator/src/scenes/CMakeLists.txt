add_library(scenes_lib 
DrawScene.cpp 
DrawSceneVk.cpp 
SceneBuilder.cpp
Scene.cpp
SimScene.cpp
MassSpringScene.cpp
MassSpringSceneGeo.cpp
TrimeshScene.cpp
cameras/CameraBase.cpp
cameras/FPSCamera.cpp
cameras/ArcBallCamera.cpp
cameras/CameraUtil.cpp
)
target_link_libraries(scenes_lib ${GLFW3_LIBRARY} ${Vulkan_LIBRARIES} )