//
// Created by Hanke on 2019-01-31.
//

#include "FPSCamera.h"
#include "CameraUtil.h"

FPSCamera::FPSCamera() : CameraBase(eCameraType::FPS_CAMERA) { Init(); }
FPSCamera::~FPSCamera() {}
FPSCamera::FPSCamera(const tVector3f &pos_, const tVector3f &centor_,
                     const tVector3f &up_)
    : CameraBase(eCameraType::FPS_CAMERA)
{
    pos = pos_;
    center = centor_;
    up = up_;
    front = centor_ - pos_;
    pitch = CameraUtil::rad2angle(asin(front(1)));
    yaw = CameraUtil::rad2angle(
        asin(asin(front(0) / cos(CameraUtil::angle2rad(pitch)))));
}

tMatrix4f FPSCamera::ViewMatrix()
{
    center = pos + front;
    return Eigen::lookAt(pos, center, up);
}

void FPSCamera::MoveForward()
{
    front.normalize();
    pos += front * key_acc;
}

void FPSCamera::MoveBackward()
{
    front.normalize();
    pos -= front * key_acc;
}

void FPSCamera::MoveLeft()
{
    tVector3f left = up.cross(front);
    left.normalize();
    pos += left * key_acc;
}

void FPSCamera::MoveRight()
{
    tVector3f left = -up.cross(front);
    left.normalize();
    pos += left * key_acc;
}

void FPSCamera::MoveUp() { pos += up * key_acc; }

void FPSCamera::MoveDown() { pos -= up * key_acc; }

void FPSCamera::MouseMove(float mouse_x, float mouse_y)
{

    if (first_mouse)
    {
        last_x = mouse_x;
        last_y = mouse_y;
        first_mouse = false;
        return;
    }
    float x_offset = mouse_x - last_x;
    float y_offset = -mouse_y + last_y;
    last_x = mouse_x;
    last_y = mouse_y;
    x_offset *= mouse_acc;
    y_offset *= mouse_acc;
    yaw += x_offset;
    pitch += y_offset;

    if (pitch > 89.0f)
        pitch = 89.0f;
    if (pitch < -89.0f)
        pitch = -89.0f;

    front(0) =
        cos(CameraUtil::angle2rad(pitch)) * sin(CameraUtil::angle2rad(yaw));
    front(1) = sin(CameraUtil::angle2rad(pitch));
    front(2) =
        -cos(CameraUtil::angle2rad(pitch)) * cos(CameraUtil::angle2rad(yaw));
    front.normalize();
}

void FPSCamera::Init()
{
    pos = tVector3f(0.8, 1, 1.8);
    center = tVector3f(0, 0, 0.2);
    up = tVector3f(0, 1, 0);
    front = center - pos;
    front.normalize();
    pitch = CameraUtil::rad2angle(asin(front(1)));
    yaw = CameraUtil::rad2angle(
        asin(asin(front(0) / cos(CameraUtil::angle2rad(pitch)))));
}
