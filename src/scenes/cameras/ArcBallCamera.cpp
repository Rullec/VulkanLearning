#include "ArcBallCamera.h"
// #include "Utils/MathUtil.h"
#include "utils/MathUtil.h"
cArcBallCamera::cArcBallCamera() : CameraBase(eCameraType::ARCBALL_CAMERA)
{
    pos = tVector3f(2, 2, 2);
    center = tVector3f(0, 0, 0);
    up = tVector3f(0, 1, 0);
    front = center - pos;
    front.normalize();
    mouse_acc *= 5e-2;
    key_acc *= 2e-2;
    // pos = tVector3f(1, 1, 0);
    // center = tVector3f(0, 1, 0);
    // up = tVector3f(0, 1, 0);
    // front = center - pos;
    // front.normalize();
}
cArcBallCamera::cArcBallCamera(const tVector3f &pos_, const tVector3f &center_,
                             const tVector3f &up_) : CameraBase(eCameraType::ARCBALL_CAMERA)
{
    pos = pos_;
    center = center_;
    up = up_;
    front = center - pos;
    front.normalize();
    mouse_acc *= 5e-2;
    key_acc *= 2e-2;
    // pos = tVector3f(1, 1, 0);
    // center = tVector3f(0, 1, 0);
    // up = tVector3f(0, 1, 0);
    // front = center - pos;
    // front.normalize();
}

cArcBallCamera::~cArcBallCamera() {}

tMatrix4f cArcBallCamera::ViewMatrix() { return Eigen::lookAt(pos, center, up); }

void cArcBallCamera::MoveForward()
{
    // decrease the dist from center to pos
    pos = (pos - center) * (1 - key_acc * 1e2) + center;
}
void cArcBallCamera::MoveBackward()
{
    // increse the dist from center to pos
    pos = (pos - center) * (1 + key_acc * 1e2) + center;
}
void cArcBallCamera::MoveLeft()
{
    // no effect
}
void cArcBallCamera::MoveRight()
{
    // no effect
}
void cArcBallCamera::MoveUp()
{
    // no effect
}
void cArcBallCamera::MoveDown()
{
    // no effect
}

/**
 * \brief           Pinned the center and rotate this arcball camera when mouse moved
*/
void cArcBallCamera::MouseMove(float mouse_x, float mouse_y)
{
    if (first_mouse)
    {
        last_x = mouse_x;
        last_y = mouse_y;
        first_mouse = false;
        return;
    }

    /*
        For screent normalized coordinate
        X,Y = (0, 0) is the left up corner
            = (1, 1) is the right down corner
        X+: from left to right
        Y+: from up to down
    */

    // 1. calculate the offset vector from last mouse pos to current mouse pos
    tVector3f offset_vec =
        tVector3f(mouse_x - last_x, -1 * (mouse_y - last_y), 0) * mouse_acc;
    last_x = mouse_x;
    last_y = mouse_y;

    // 2. convert this vector to world frame, and opposite it (because we want to rotate the object indeed)
    tVector3f offset_vec_world =
        -1 * ViewMatrix().block(0, 0, 3, 3).inverse() * offset_vec;

    // 3. calcualte center_to_pos vector, calculate the rotmat for our camera (fixed center)
    tVector3f center_to_pos = pos - center;

    tMatrix3f rotmat =
        cMathUtil::AxisAngleToRotmat(
            cMathUtil::Expand(
                center_to_pos.normalized().cross(offset_vec_world), 0))
            .block(0, 0, 3, 3)
            .cast<float>();

    center_to_pos = rotmat.cast<float>() * center_to_pos;

    // 4. rotate the center to pos, update other variables, center is fixed
    up = rotmat.cast<float>() * this->up;
    pos = center_to_pos + center;
    front = (center - pos).normalized();
}