#pragma once
#include "utils/MathUtil.h"

class CameraUtil
{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    static float angle2rad(double angle)
    {
        return static_cast<float>(M_PI * angle / 180.0);
    }

    static float rad2angle(double rad)
    {
        return static_cast<float>(rad / M_PI * 180.0);
    }
    static tVector3f pos;
    static tVector3f center;
    static tVector3f up;
    static tVector3f front;
    static float pitch, yaw;
};
