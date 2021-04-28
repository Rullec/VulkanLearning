// #ifdef USE_OPTIX
#pragma once

#include "utils/MathUtil.h"
#include "optix7.h"

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

    OptixTraversableHandle traversable;
};

// #endif
