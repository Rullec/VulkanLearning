add_library(scenes_lib 
DrawScene.cpp 
DrawSceneVk.cpp 
SceneBuilder.cpp
Scene.cpp
SimScene.cpp
SemiImplicitScene.cpp
ImplicitScene.cpp
FastSimScene.cpp
PBDScene.cpp
cameras/CameraBase.cpp
cameras/FPSCamera.cpp
cameras/ArcBallCamera.cpp
cameras/CameraUtil.cpp
)
target_link_libraries(scenes_lib ${GLFW3_LIBRARY} ${Vulkan_LIBRARIES} )