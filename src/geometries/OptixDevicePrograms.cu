
// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "OptixLaunchParam.h"
#include "utils/MathUtil.h"
#include <optix_device.h>

/*! launch parameters in constant memory, filled in by optix upon
      optixLaunch (this gets filled in from the buffer we pass to
      optixLaunch) */
extern "C" __constant__ LaunchParams optixLaunchParams;

// for this simple example, we have a single ray type
enum
{
    SURFACE_RAY_TYPE = 0,
    RAY_TYPE_COUNT
};

//------------------------------------------------------------------------------
// closest hit and anyhit programs for radiance-type rays.
//
// Note eventually we will have to create one pair of those for each
// ray type and each geometry type we want to render; but this
// simple example doesn't use any actual geometries yet, so we only
// create a single, dummy, set of them (we do have to have at least
// one group of them to set up the SBT)
//------------------------------------------------------------------------------

// tVector3f randomColor(int i)
// {

// }
extern "C" __global__ void __closesthit__radiance()
{
    const int primID = optixGetPrimitiveIndex();

    // prd = gdt::randomColor(primID);
    // prd = randomColor(primID);
    // int r = unsigned(primID) * 13 * 17 + 0x234235;
    // int g = unsigned(primID) * 7 * 3 * 5 + 0x773477;
    // int b = unsigned(primID) * 11 * 19 + 0x223766;
    // int t = optixGetRayTmax() * 400;

    // {
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;
    tVector3f rayDir =
        ((optixLaunchParams.convert_mat * tVector4f(ix, iy, 1, 1))
             .segment(0, 3) -
         optixLaunchParams.position)
            .normalized() *
        optixGetRayTmax();
    // 1. get camera pos, get ray direction
    // float3 ray_dir = make_float3(
    //         rayDir.x(),
    //         rayDir.y(),
    //         rayDir.z());
    // ray_dir.x = ray_dir.x ;
    // ray_dir.y = ray_dir.y * optixGetRayTmax();
    // ray_dir.z = ray_dir.z * optixGetRayTmax();

    // 2. get the normal vector of the camera plane
    tVector3f eigen_focus_dir =
        (optixLaunchParams.camera_center - optixLaunchParams.position)
            .normalized();
    // float3 focus_dir = make_float3(
    //     eigen_focus_dir.x(),
    //     eigen_focus_dir.y(),
    //     eigen_focus_dir.z()
    // );
    float real_depth = std::fabs(eigen_focus_dir.dot(rayDir));
    // int t = real_depth;
    uint32_t t = *(reinterpret_cast<uint32_t *>(&real_depth));
    // int t = static_cast<int>(real_depth * 200);
    // t = t > 255 ? 255 : t;
    // t = 255;
    // }
    // 3. dot product then abs, get the real depth
    optixSetPayload_0(t);
}

extern "C" __global__ void __anyhit__radiance()
{ /*! for this simple example, this will remain empty */
}

//------------------------------------------------------------------------------
// miss program that gets called for any ray that did not have a
// valid intersection
//
// as with the anyhit/closest hit programs, in this example we only
// need to have _some_ dummy function to set up a valid SBT
// ------------------------------------------------------------------------------

extern "C" __global__ void __miss__radiance()
{
    // tVector3f &prd = *(tVector3f*)getPRD<tVector3f>();
    // // set to constant white as background color
    // prd = tVector3f(1.f, 1.f, 1.f);
    optixSetPayload_0(0);
}

//------------------------------------------------------------------------------
// ray gen program - the actual rendering happens in here
//------------------------------------------------------------------------------
extern "C" __global__ void __raygen__renderFrame()
{
    // compute a test pattern based on pixel ID
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;

    // const auto &camera = optixLaunchParams.camera;

    // our per-ray data for this example. what we initialize it to
    // won't matter, since this value will be overwritten by either
    // the miss or hit program, anyway

    // the values we store the PRD pointer in:
    uint32_t d;

    // normalized screen plane position, in [0,1]^2
    // const tVector2f
    // screen(tVector2f(ix+.5f,iy+.5f).cwiseQuotient(optixLaunchParams.frame.size.cast<float>()));

    // generate ray direction
    /*
    tVector3f target_pixel_world_pos = (optixLaunchParams.convert_mat *
    tVector4f(ix, iy, 1, 1)).segment(0, 3); tVector3f rayDir =
    (target_pixel_world_pos - camera.position).normalized();
    */
    // tVector3f rayDir = (camera.direction
    //                          + (float(screen.x()) - 0.5f) * camera.horizontal
    //                          + (float(screen.y()) - 0.5f) *
    //                          camera.vertical).normalized();
    tVector3f rayDir =
        ((optixLaunchParams.convert_mat * tVector4f(ix, iy, 1, 1))
             .segment(0, 3) -
         optixLaunchParams.position)
            .normalized();
    // tVector3f rayDir = ( optixLaunchParams.convert_mat * tVector4f(ix, iy, 1,
    // 1)).segment(0, 3); rayDir = optixLaunchParams.convert_mat.block(0, 0, 3,
    // 3)
    // * rayDir;

    float3 new_dir = make_float3(rayDir.x(), rayDir.y(), rayDir.z());
    float3 new_position = make_float3(optixLaunchParams.position.x(),
                                      optixLaunchParams.position.y(),
                                      optixLaunchParams.position.z());
    
    if(
        (ix < optixLaunchParams.raycast_range(0,0) || ix > optixLaunchParams.raycast_range(0, 1))
        ||
        (iy < optixLaunchParams.raycast_range(1,0) || iy > optixLaunchParams.raycast_range(1,1))
    )
    {
        d = 0;
    }
    else {
        optixTrace(optixLaunchParams.traversable, new_position, new_dir,
            0.f,   // tmin
            1e20f, // tmax
            0.0f,  // rayTime
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT, // OPTIX_RAY_FLAG_NONE,
            SURFACE_RAY_TYPE,              // SBT offset
            RAY_TYPE_COUNT,                // SBT stride
            SURFACE_RAY_TYPE,              // missSBTIndex
            d);
    }

                     

    // const int int_r = int(255.99f*__int_as_float(r));
    // const int int_g = int(255.99f*__int_as_float(g));
    // const int int_b = int(255.99f*__int_as_float(b));
    // const int int_r = d;
    // const int int_g = d;
    // const int int_b = d;

    // convert to 32-bit rgba value (we explicitly set alpha to 0xff
    // to make stb_image_write happy ...
    // const uint32_t rgba =
    //     0xff000000 | (int_r << 0) | (int_g << 8) | (int_b << 16);
    
    // and write to frame buffer ...
    const uint32_t fbIndex = ix + iy * optixLaunchParams.frame.size.x();
    optixLaunchParams.frame.colorBuffer[fbIndex] = d;
}
