#ifdef USE_OPTIX
#pragma once
#include "Raycaster.h"
#include "OptixCUDABuffer.h"
#include "OptixLaunchParam.h"

class cOptixRaycaster : public cRaycaster
{
public:
    explicit cOptixRaycaster();
    virtual void AddResources(const std::vector<tTriangle *> triangles,
                              const std::vector<tVertex *> vertices) override;
    virtual void CalcDepthMap(int height, int width, CameraBasePtr camera, std::string path) override final;
    virtual void CalcDepthMapMultiCamera(int height, int width, std::vector<CameraBasePtr> camera_array, std::vector<std::string> path_array);

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
    void UpdateVertexBufferToCuda();
    // -----------vars-------------
    int cur_width, cur_height;
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
};

#define GDT_TERMINAL_RED "\033[1;31m"
#define GDT_TERMINAL_GREEN "\033[1;32m"
#define GDT_TERMINAL_YELLOW "\033[1;33m"
#define GDT_TERMINAL_BLUE "\033[1;34m"
#define GDT_TERMINAL_RESET "\033[0m"
#define GDT_TERMINAL_DEFAULT GDT_TERMINAL_RESET
#define GDT_TERMINAL_BOLD "\033[1;1m"

#endif // USE_OPTIX