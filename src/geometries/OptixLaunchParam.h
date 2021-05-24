// #ifdef USE_OPTIX
#pragma once

#include "optix7.h"
#include "utils/MathUtil.h"

struct MY_ALIGN(16) LaunchParams
{
    struct MY_ALIGN(16)
    {
        uint32_t *colorBuffer;
        tVector2i size;
    } frame;

    // struct
    // {
    MY_ALIGN(16)
    tVector3f position;
    // tVector3f direction;
    // MY_ALIGN(16)
    // tVector3f horizontal;
    // MY_ALIGN(16)
    // tVector3f vertical;
    // } camera;

    MY_ALIGN(16)
    tMatrix4f convert_mat;

    MY_ALIGN(16)
    tVector3f camera_up;
    tVector3f camera_center;

    OptixTraversableHandle traversable;
};

// #endif
