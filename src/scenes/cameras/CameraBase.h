#pragma once
//
// Extract to base by Xudong on 2021-01-30
//

// #include "../Utils/EigenUtils.h"
#include "utils/MathUtil.h"
#include <iostream>
// #include <cmath>

enum eCameraType
{
    FPS_CAMERA = 0, // default wasd FPS camear
    ARCBALL_CAMERA, // arcball camera
    NUM_OF_CAMERA_TYPE
};

class CameraBase
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    CameraBase(eCameraType type);
    virtual ~CameraBase() = 0;
    eCameraType GetType() const;
    virtual tMatrix4f ViewMatrix() = 0;

    virtual void MoveForward() = 0;
    virtual void MoveBackward() = 0;
    virtual void MoveLeft() = 0;
    virtual void MoveRight() = 0;
    virtual void MoveUp() = 0;
    virtual void MoveDown() = 0;

    virtual void MouseMove(float mouse_x, float mouse_y) = 0;
    virtual void SetXY(float mouse_x, float mouse_y);
    virtual void ResetFlag();

    virtual void SetKeyAcc(double acc);
    virtual void SetMouseAcc(double acc);
    virtual void SetStatus();

    float pitch, yaw;
    tVector3f pos, center, up, front;
    float fov;

protected:
    eCameraType type;
    float mouse_acc;
    float key_acc;

    float last_x, last_y; // the previous x and y position of mouse
    bool first_mouse;     // whether it's the first mouse event
};