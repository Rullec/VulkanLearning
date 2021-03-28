add_library(scenes_lib 
DrawScene.cpp 
CameraBase.cpp
FPSCamera.cpp
ArcBallCamera.cpp
)
target_link_libraries(scenes_lib ${GLFW3_LIBRARY})