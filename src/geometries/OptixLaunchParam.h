// #ifdef USE_OPTIX
#pragma once

#include "Optix7.h"
#include "utils/MathUtil.h"
#define OPTIX_LAUNCH_PARAM_NUM_OF_RANDOM_NUMBER 10000
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

    MY_ALIGN(16)
    Eigen::Matrix2i raycast_range;

    MY_ALIGN(16)
    Eigen::Matrix4i
        start_triangle_id_for_each_object; // start triangle id for each object.
                                           // used for object identification
    MY_ALIGN(16)
    uint8_t num_of_objects;

    MY_ALIGN(16)
    int random_seed; // noise random seed: current disabled

    MY_ALIGN(16)
    Eigen::Matrix4i
        disable_raycast_for_objects; // switch to enalbe raycast. if the value is
                                    // zero, the depth will be set to zero.

    MY_ALIGN(16)
    OptixTraversableHandle traversable;
};

// #endif
