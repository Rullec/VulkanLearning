#ifdef USE_OPTIX
#pragma once
#include "OptixCUDABuffer.h"
#include "OptixLaunchParam.h"
#include "Raycaster.h"

namespace Json
{
class Value;
};
class cOptixRaycaster : public cRaycaster
{
public:
    explicit cOptixRaycaster(bool enable_only_exporting_cutted_window);
    // virtual void AddResources(const std::vector<tTriangle *> triangles,
    //                           const std::vector<tVertex *> vertices)
    //                           override;
    virtual void AddResources(cBaseObjectPtr object);
    virtual void CalcDepthMap(const tMatrix2i &cast_range, int height,
                              int width, CameraBasePtr camera,
                              std::string path) override final;
    virtual void
    CalcDepthMapMultiCamera(const tMatrix2i &cast_range, int height, int width,
                            std::vector<CameraBasePtr> camera_array,
                            std::vector<std::string> path_array);
    virtual void CalcDepthMapMultiCamera(const tMatrix2i &cast_range,
                                         int height, int width,
                                         CameraBasePtr cur_cam,
                                         tVectorXf &pixels);

protected:
    // -----------methods------------
    void InitOptix();
    void CreateContext();
    void CreateModule();
    void CreateRaygenPrograms();
    void CreateMissPrograms();
    void CreateHitgroupPrograms();
    void Rebuild();
    void CreatePipeline();
    void BuildSBT();
    OptixTraversableHandle BuildAccel();
    void BuildGeometryCudaHostBuffer();
    void resize(const tVector2i &newSize);
    void setCamera(const CameraBasePtr &camera);
    void render();
    void downloadPixels(uint32_t h_pixels[]);
    void downloadPixels(float *h_pixels);
    void UpdateVertexBufferToCuda();
    // -----------vars-------------
    int camera_width, camera_height; // the raycasting full view resolution

    int visible_window_width,
        visible_window_height; // we may only be insterested in a small fraction
                               // window of the whole picture
    tVector2i visible_window_st; // [width_st, height_st]
    CUcontext cudaContext;
    CUstream stream;
    cudaDeviceProp deviceProps;
    CameraBasePtr lastSetCamera;

    //! the optix context that our pipeline will run in.
    OptixDeviceContext optixContext;

    OptixPipeline pipeline;
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    OptixPipelineLinkOptions pipelineLinkOptions = {};

    OptixModule module;
    OptixModuleCompileOptions moduleCompileOptions = {};

    std::vector<OptixProgramGroup> raygenPGs;
    CUDABuffer raygenRecordsBuffer;
    std::vector<OptixProgramGroup> missPGs;
    CUDABuffer missRecordsBuffer;
    std::vector<OptixProgramGroup> hitgroupPGs;
    CUDABuffer hitgroupRecordsBuffer;
    OptixShaderBindingTable sbt = {};

    LaunchParams launchParams;
    CUDABuffer launchParamsBuffer;

    CUDABuffer colorBuffer;
    CUDABuffer vertexBuffer;
    CUDABuffer indexBuffer;
    //! buffer that keeps the (final, compacted) accel structure
    CUDABuffer asBuffer;

    std::vector<tVector3f> cuda_host_vertices_buffer;
    std::vector<tVector3i> cuda_host_index_buffer;

    bool SavePngDepthImage(const std::vector<float> &pixels, const char *path);
};

#define GDT_TERMINAL_RED "\033[1;31m"
#define GDT_TERMINAL_GREEN "\033[1;32m"
#define GDT_TERMINAL_YELLOW "\033[1;33m"
#define GDT_TERMINAL_BLUE "\033[1;34m"
#define GDT_TERMINAL_RESET "\033[0m"
#define GDT_TERMINAL_DEFAULT GDT_TERMINAL_RESET
#define GDT_TERMINAL_BOLD "\033[1;1m"

#endif // USE_OPTIX