add_library(scenes_lib 
DrawScene.cpp 
DrawSceneVk.cpp 
CameraBase.cpp
FPSCamera.cpp
ArcBallCamera.cpp
SceneBuilder.cpp
Scene.cpp
SimScene.cpp
CameraUtil.cpp
)
target_link_libraries(scenes_lib ${GLFW3_LIBRARY})