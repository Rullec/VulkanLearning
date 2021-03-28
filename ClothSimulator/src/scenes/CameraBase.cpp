#include "CameraBase.h"

#include "CameraUtil.h"
CameraBase::CameraBase(eCameraType type_)
    : type(type_), mouse_acc(0.1f), key_acc(0.02f), last_x(0), last_y(0),
      first_mouse(true)
{
    pitch = 0;
    yaw = 0;
    pos.setZero();
    center = tVector3f(0, 0, 0);
    pos = tVector3f(1, 0, 0);
    front = tVector3f(-1, 0, 0);
}
CameraBase::~CameraBase() {}
eCameraType CameraBase::GetType() const { return type; }

void CameraBase::SetKeyAcc(double acc) { key_acc = static_cast<float>(acc); }

void CameraBase::SetMouseAcc(double acc)
{
    mouse_acc = static_cast<float>(acc);
}

void CameraBase::SetXY(float mouse_x, float mouse_y)
{
    last_x = mouse_x;
    last_y = mouse_y;
}
void CameraBase::ResetFlag() { first_mouse = true; }
void CameraBase::SetStatus()
{

    this->pos = CameraUtil::pos;
    this->center = CameraUtil::center;
    this->up = CameraUtil::up;
    this->front = CameraUtil::front;
    this->pitch = CameraUtil::pitch;
    this->yaw = CameraUtil::yaw;
}