//
// Created by Xudong on 2021/01/30
//
#pragma once
#include "scenes/CameraBase.h"

/**
 * \brief           Arcball camera
*/
class ArcBallCamera : public CameraBase
{
public:
    ArcBallCamera();
    ArcBallCamera(const tVector3f &pos, const tVector3f &centor,
                  const tVector3f &up);

    virtual ~ArcBallCamera();
    virtual tMatrix4f ViewMatrix() override;

    virtual void MoveForward() override;
    virtual void MoveBackward() override;
    virtual void MoveLeft() override;
    virtual void MoveRight() override;
    virtual void MoveUp() override;
    virtual void MoveDown() override;

    virtual void MouseMove(float mouse_x, float mouse_y) override;

protected:
};