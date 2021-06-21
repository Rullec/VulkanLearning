#ifdef USE_OPTIX
#include "OptixRaycaster.h"
#include "optix_function_table_definition.h"
#include "scenes/cameras/CameraBase.h"
#include "utils/JsonUtil.h"
#include "utils/LogUtil.h"
#include <iostream>

extern "C" char embedded_ptx_code[];

/*! SBT record for a raygen program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord
{
    __align__(
        OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    // data here
    void *data;
};

/*! SBT record for a miss program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord
{
    __align__(
        OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    // data here
    void *data;
};

/*! SBT record for a hitgroup program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord
{
    __align__(
        OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    // data here
    int objectID;
};

// cOptixRaycaster::cOptixRaycaster(const std::vector<tTriangle *> triangles,
//                                  const std::vector<tVertex *> vertices) :
//                                  cRaycaster(triangles, vertices)
cOptixRaycaster::cOptixRaycaster() { mEnableOnlyExportCuttedWindow = false; }

void cOptixRaycaster::Init(const Json::Value &conf)
{
    cRaycaster::Init(conf);
    mEnableOnlyExportCuttedWindow =
        cJsonUtil::ParseAsBool(ENABLE_ONLY_EXPORTING_CUTTED_WINDOW_KEY, conf);

    InitEnableDepthForObjects(conf);

    InitOptix();

    // std::cout << "creating optix context ..." << std::endl;
    CreateContext();

    // std::cout << "setting up module ..." << std::endl;
    CreateModule();

    // std::cout << "creating raygen programs ..." << std::endl;
    CreateRaygenPrograms();
    // std::cout << "creating miss programs ..." << std::endl;
    CreateMissPrograms();
    // std::cout << "creating hitgroup programs ..." << std::endl;
    CreateHitgroupPrograms();
}
// void cOptixRaycaster::AddResources(const std::vector<tTriangle *> triangles,
//                                    const std::vector<tVertex *> vertices)
#include "sim/BaseObject.h"
void cOptixRaycaster::AddResources(cBaseObjectPtr object)
{
    cRaycaster::AddResources(object);

    // check the name
    {
        std::string obj_name = object->GetObjName();
        std::map<std::string, bool>::iterator it =
            disable_depth_for_objects.find(obj_name);
        if (it != disable_depth_for_objects.end())
        {
            std::cout << "[log] raycaster add obj " << obj_name
                      << ", disable raycast depth = " << it->second
                      << std::endl;
            int cur_id = mObjects.size() - 1;
            // std::cout << "cur id = " << cur_id << std::endl;
            launchParams.disable_raycast_for_objects(cur_id / 4, cur_id % 4) =
                it->second;
        }

        // exit(1);
    }

    BuildStartTriangleIdForObjects();

    launchParams.traversable = BuildAccel();
    // auto a = <=>(1, 2);
    // std::cout << a <<std::endl;
    // exit(1);

    // std::cout << "setting up optix pipeline ..." << std::endl;
    CreatePipeline();

    // std::cout << "building SBT ..." << std::endl;
    BuildSBT();

    launchParamsBuffer.alloc(sizeof(launchParams));

    // // std::cout << "sizeof launchparam = " << sizeof(launchParams) <<
    // std::endl; exit(0); std::cout << "context, module, pipeline, etc, all set
    // up ..." << std::endl;

    // std::cout << GDT_TERMINAL_GREEN;
    // std::cout << "Optix 7 Sample fully set up" << std::endl;
    // std::cout << GDT_TERMINAL_DEFAULT;
}
void cOptixRaycaster::Rebuild()
{
    launchParams.traversable = BuildAccel();
    // std::cout << "setting up optix pipeline ..." << std::endl;
    // CreatePipeline();

    // std::cout << "building SBT ..." << std::endl;
    // BuildSBT();
    launchParamsBuffer.alloc(sizeof(launchParams));
}

void cOptixRaycaster::InitOptix()
{
    // std::cout << "initializing optix..." << std::endl;

    // -------------------------------------------------------
    // check for available optix7 capable devices
    // -------------------------------------------------------
    cudaFree(0);
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    if (numDevices == 0)
        throw std::runtime_error("no CUDA capable devices found!");
    // std::cout << "found " << numDevices << " CUDA devices" << std::endl;

    // -------------------------------------------------------
    // initialize optix
    // -------------------------------------------------------
    OPTIX_CHECK(optixInit());
    // std::cout << GDT_TERMINAL_GREEN
    //           << "successfully initialized optix... yay!"
    //           << GDT_TERMINAL_DEFAULT << std::endl;
}

static void context_log_cb(unsigned int level, const char *tag,
                           const char *message, void *)
{
    printf("[%2d][%12s]: %s\n", (int)level, tag, message);
}

/*! creates and configures a optix device context (in this simple
      example, only for the primary GPU device) */
void cOptixRaycaster::CreateContext()
{
    // for this sample, do everything on one device
    const int deviceID = 0;
    CUDA_CHECK(SetDevice(deviceID));
    CUDA_CHECK(StreamCreate(&stream));

    cudaGetDeviceProperties(&deviceProps, deviceID);
    // std::cout << "running on device: " << deviceProps.name << std::endl;

    CUresult cuRes = cuCtxGetCurrent(&cudaContext);
    if (cuRes != CUDA_SUCCESS)
        printf("Error querying current context: error code %d\n", cuRes);

    OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
    OPTIX_CHECK(optixDeviceContextSetLogCallback(optixContext, context_log_cb,
                                                 nullptr, 3));
}

/*! creates the module that contains all the programs we are going
      to use. in this simple example, we use a single module from a
      single .cu file, using a single embedded ptx string */
void cOptixRaycaster::CreateModule()
{
    moduleCompileOptions.maxRegisterCount = 50;
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    pipelineCompileOptions = {};
    pipelineCompileOptions.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.numPayloadValues = 4;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName =
        "optixLaunchParams";

    pipelineLinkOptions.maxTraceDepth = 2;

    const std::string ptxCode = embedded_ptx_code;

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixModuleCreateFromPTX(
        optixContext, &moduleCompileOptions, &pipelineCompileOptions,
        ptxCode.c_str(), ptxCode.size(), log, &sizeof_log, &module));
    if (sizeof_log > 1)
        std::cout << log;
}

/*! does all setup for the raygen program(s) we are going to use */
void cOptixRaycaster::CreateRaygenPrograms()
{
    // we do a single ray gen program in this example:
    raygenPGs.resize(1);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module = module;
    pgDesc.raygen.entryFunctionName = "__raygen__renderFrame";

    // OptixProgramGroup raypg;
    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(optixContext, &pgDesc, 1, &pgOptions,
                                        log, &sizeof_log, &raygenPGs[0]));
    if (sizeof_log > 1)
        std::cout << log;
}

/*! does all setup for the miss program(s) we are going to use */
void cOptixRaycaster::CreateMissPrograms()
{
    // we do a single ray gen program in this example:
    missPGs.resize(1);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.miss.module = module;
    pgDesc.miss.entryFunctionName = "__miss__radiance";

    // OptixProgramGroup raypg;
    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(optixContext, &pgDesc, 1, &pgOptions,
                                        log, &sizeof_log, &missPGs[0]));
    if (sizeof_log > 1)
        std::cout << log;
}

/*! does all setup for the hitgroup program(s) we are going to use */
void cOptixRaycaster::CreateHitgroupPrograms()
{
    // for this simple example, we set up a single hit group
    hitgroupPGs.resize(1);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.hitgroup.moduleCH = module;
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    pgDesc.hitgroup.moduleAH = module;
    pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(optixContext, &pgDesc, 1, &pgOptions,
                                        log, &sizeof_log, &hitgroupPGs[0]));
    if (sizeof_log > 1)
        std::cout << log;
}

/*! assembles the full pipeline of all programs */
void cOptixRaycaster::CreatePipeline()
{
    std::vector<OptixProgramGroup> programGroups;
    for (auto pg : raygenPGs)
        programGroups.push_back(pg);
    for (auto pg : missPGs)
        programGroups.push_back(pg);
    for (auto pg : hitgroupPGs)
        programGroups.push_back(pg);

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(optixContext, &pipelineCompileOptions,
                                    &pipelineLinkOptions, programGroups.data(),
                                    (int)programGroups.size(), log, &sizeof_log,
                                    &pipeline));
    // if (sizeof_log > 1)
    //     std::cout << log;

    OPTIX_CHECK(
        optixPipelineSetStackSize(/* [in] The pipeline to configure the stack
                                     size for */
                                  pipeline,
                                  /* [in] The direct stack size requirement for
            direct callables invoked from IS or AH. */
                                  2 * 1024,
                                  /* [in] The direct stack size requirement for
            direct callables invoked from RG, MS, or CH.  */
                                  2 * 1024,
                                  /* [in] The continuation stack requirement. */
                                  2 * 1024,
                                  /* [in] The maximum depth of a traversable
            graph passed to trace. */
                                  1));
    // if (sizeof_log > 1)
    //     std::cout << log;
}

/*! constructs the shader binding table */
void cOptixRaycaster::BuildSBT()
{
    // ------------------------------------------------------------------
    // build raygen records
    // ------------------------------------------------------------------
    std::vector<RaygenRecord> raygenRecords(0);
    for (int i = 0; i < raygenPGs.size(); i++)
    {
        RaygenRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGs[i], &rec));
        rec.data = nullptr; /* for now ... */
        raygenRecords.push_back(rec);
    }
    raygenRecordsBuffer.alloc_and_upload(raygenRecords);
    sbt.raygenRecord = raygenRecordsBuffer.d_pointer();

    // ------------------------------------------------------------------
    // build miss records
    // ------------------------------------------------------------------
    std::vector<MissRecord> missRecords(0);
    for (int i = 0; i < missPGs.size(); i++)
    {
        MissRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[i], &rec));
        rec.data = nullptr; /* for now ... */
        missRecords.push_back(rec);
    }
    missRecordsBuffer.alloc_and_upload(missRecords);
    sbt.missRecordBase = missRecordsBuffer.d_pointer();
    sbt.missRecordStrideInBytes = sizeof(MissRecord);
    sbt.missRecordCount = (int)missRecords.size();

    // ------------------------------------------------------------------
    // build hitgroup records
    // ------------------------------------------------------------------

    // we don't actually have any objects in this example, but let's
    // create a dummy one so the SBT doesn't have any null pointers
    // (which the sanity checks in compilation would complain about)
    int numObjects = 1;
    std::vector<HitgroupRecord> hitgroupRecords(0);
    for (int i = 0; i < numObjects; i++)
    {
        int objectType = 0;
        HitgroupRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[objectType], &rec));
        rec.objectID = i;
        hitgroupRecords.push_back(rec);
    }
    hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
    sbt.hitgroupRecordBase = hitgroupRecordsBuffer.d_pointer();
    sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    sbt.hitgroupRecordCount = (int)hitgroupRecords.size();
}

/**
 * \brief           set the start triangle id for each object
 */
void cOptixRaycaster::BuildStartTriangleIdForObjects()
{
    launchParams.start_triangle_id_for_each_object.setZero();
    launchParams.num_of_objects = mObjects.size();

    SIM_ASSERT(mObjects.size() < 16);

    int offset = 0;
    for (int i = 0; i < mObjects.size(); i++)
    {
        launchParams.start_triangle_id_for_each_object(i / 4, i % 4) = offset;
        // printf("[log] obj %d offset %d\n", i, offset);
        offset += mTriangleArray_lst[i].size();
    }
    // exit(1);
}

void cOptixRaycaster::UpdateVertexBufferToCuda()
{
    BuildAccel();
    std::cout << "upload the verteix buffer to cuda\n";
}
/**
 * \brief           Build the acceleration strucutre
 */
OptixTraversableHandle cOptixRaycaster::BuildAccel()
{
    BuildGeometryCudaHostBuffer();
    vertexBuffer.alloc_and_upload(cuda_host_vertices_buffer);
    indexBuffer.alloc_and_upload(cuda_host_index_buffer);
    OptixTraversableHandle asHandle{0};

    // ==================================================================
    // triangle inputs
    // ==================================================================
    OptixBuildInput triangleInput = {};
    triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    // create local variables, because we need a *pointer* to the
    // device pointers
    CUdeviceptr d_vertices = vertexBuffer.d_pointer();
    CUdeviceptr d_indices = indexBuffer.d_pointer();

    // it's the cuda format, near the same as mine
    triangleInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangleInput.triangleArray.vertexStrideInBytes = sizeof(tVector3f);
    triangleInput.triangleArray.numVertices =
        (int)cuda_host_vertices_buffer.size();
    triangleInput.triangleArray.vertexBuffers = &d_vertices;

    triangleInput.triangleArray.indexFormat =
        OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangleInput.triangleArray.indexStrideInBytes = sizeof(tVector3i);
    triangleInput.triangleArray.numIndexTriplets =
        (int)cuda_host_index_buffer.size();
    triangleInput.triangleArray.indexBuffer = d_indices;

    uint32_t triangleInputFlags[1] = {0};

    // in this example we have one SBT entry, and no per-primitive
    // materials:
    triangleInput.triangleArray.flags = triangleInputFlags;
    triangleInput.triangleArray.numSbtRecords = 1;
    triangleInput.triangleArray.sbtIndexOffsetBuffer = 0;
    triangleInput.triangleArray.sbtIndexOffsetSizeInBytes = 0;
    triangleInput.triangleArray.sbtIndexOffsetStrideInBytes = 0;

    // ==================================================================
    // BLAS setup
    // ==================================================================

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags =
        OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accelOptions.motionOptions.numKeys = 1;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes blasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(optixContext, &accelOptions,
                                             &triangleInput,
                                             1, // num_build_inputs
                                             &blasBufferSizes));

    // ==================================================================
    // prepare compaction
    // ==================================================================

    CUDABuffer compactedSizeBuffer;
    compactedSizeBuffer.alloc(sizeof(uint64_t));

    OptixAccelEmitDesc emitDesc;
    emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = compactedSizeBuffer.d_pointer();

    // ==================================================================
    // execute build (main stage)
    // ==================================================================

    CUDABuffer tempBuffer;
    tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);

    CUDABuffer outputBuffer;
    outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

    OPTIX_CHECK(
        optixAccelBuild(optixContext,
                        /* stream */ 0, &accelOptions, &triangleInput, 1,
                        tempBuffer.d_pointer(), tempBuffer.sizeInBytes,

                        outputBuffer.d_pointer(), outputBuffer.sizeInBytes,

                        &asHandle,

                        &emitDesc, 1));
    CUDA_SYNC_CHECK();

    // ==================================================================
    // perform compaction
    // ==================================================================
    uint64_t compactedSize;
    compactedSizeBuffer.download(&compactedSize, 1);

    asBuffer.alloc(compactedSize);
    OPTIX_CHECK(optixAccelCompact(optixContext,
                                  /*stream:*/ 0, asHandle, asBuffer.d_pointer(),
                                  asBuffer.sizeInBytes, &asHandle));
    CUDA_SYNC_CHECK();

    // ==================================================================
    // aaaaaand .... clean up
    // ==================================================================
    outputBuffer.free(); // << the UNcompacted, temporary output buffer
    tempBuffer.free();
    compactedSizeBuffer.free();

    return asHandle;
}

void cOptixRaycaster::BuildRandomSeries()
{
    launchParams.random_seed = cMathUtil::RandInt();
}
/**
 * \brief           Build the geometry host buffer
 */
void cOptixRaycaster::BuildGeometryCudaHostBuffer()
{
    // std::cout << "[debug] recalculate the cuda buffer\n";
    // 1. get the total size of vertices and triangles buffer
    int vertices_size = 0, triangle_size = 0;
    for (int i = 0; i < mTriangleArray_lst.size(); i++)
    {
        vertices_size += mVertexArray_lst[i].size();
        triangle_size += mTriangleArray_lst[i].size();
    }
    cuda_host_vertices_buffer.resize(vertices_size, tVector3f::Zero());
    if (cuda_host_index_buffer.size() != triangle_size)
        cuda_host_index_buffer.resize(triangle_size, tVector3i::Zero());

    int num_of_obj = mTriangleArray_lst.size();
    int cur_obj_verteix_offset = 0;
    int total_triangle_idx = 0, total_vertex_idx = 0;
    for (int obj_id = 0; obj_id < num_of_obj; obj_id++)
    {
        // 1. add this object's indices and vertex data
        auto tri_array = mTriangleArray_lst[obj_id];
        auto ver_array = mVertexArray_lst[obj_id];
        for (int i = 0; i < tri_array.size(); i++)
        {
            cuda_host_index_buffer[total_triangle_idx++].noalias() =
                tVector3i(tri_array[i]->mId0 + cur_obj_verteix_offset,
                          tri_array[i]->mId1 + cur_obj_verteix_offset,
                          tri_array[i]->mId2 + cur_obj_verteix_offset);
        }
        // 2. add the bias

        for (int i = 0; i < ver_array.size(); i++)
        {
            cuda_host_vertices_buffer[total_vertex_idx++].noalias() =
                ver_array[i]->mPos.segment(0, 3).cast<float>();
        }

        cur_obj_verteix_offset += ver_array.size();
    }
    // int st = 0;
    // for (auto &mTriangleArray : mTriangleArray_lst)
    // {
    //     for (int i = 0; i < mTriangleArray.size(); i++)
    //     {
    //         cuda_host_index_buffer[st].noalias() =
    //             tVector3i(
    //                 mTriangleArray.at(i)->mId0,
    //                 mTriangleArray.at(i)->mId1,
    //                 mTriangleArray.at(i)->mId2);
    //         st++;
    //     }
    // }

    // st = 0;
    // for (auto &mVertexArray : mVertexArray_lst)
    // {
    //     for (int i = 0; i < mVertexArray.size(); i++)
    //     {
    //         cuda_host_vertices_buffer[st++].noalias() =
    //             tVector3f(
    //                 mVertexArray.at(i)->mPos[0],
    //                 mVertexArray.at(i)->mPos[1],
    //                 mVertexArray.at(i)->mPos[2]);
    //     }
    // }
    // std::cout << "triangle size = " << mTriangleArray.size() << std::endl;
    // std::cout << "vertices size = " << mVertexArray->size() << std::endl;
}

// void save(const std::string fileName, int width, int height,
//           uint32_t h_pixels[])
// {
//     stbi_write_png(fileName.c_str(), width, height, 4, h_pixels,
//                    width * sizeof(uint32_t));
// }
/**
 * \brief           Calculate the depth image
 */
extern bool SavePNGSingleChannel(const float *depth_pixels, int width,
                                 int height, const char *outfile_name);
void cOptixRaycaster::CalcDepthMap(const tMatrix2i &cast_range,
                                   int _camera_height, int _camera_width,
                                   CameraBasePtr camera, std::string path)
{
    camera_width = _camera_width;
    camera_height = _camera_height;
    CalcCastWindowSize(cast_range, visible_window_width, visible_window_height,
                       visible_window_st);

    launchParams.raycast_range = cast_range;
    Rebuild();

    // 1. if the size is changed, we need to resize the buffer
    setCamera(camera);
    if (launchParams.frame.size.x() != camera_width ||
        launchParams.frame.size.y() != camera_height)
    {
        tVector2i size = tVector2i(camera_width, camera_height);
        resize(size);
    }
    // std::cout << "resize done\n";

    // UpdateVertexBufferToCuda();
    // 2. reupload all values in launch param
    render();
    // 3. launch the optix program
    // std::cout << "render done\n";
    std::vector<float> pixels(camera_width * camera_height, 0);
    this->downloadPixels(pixels.data());
    // std::cout << "download pixels done\n";

    SavePngDepthImage(pixels, path.c_str());
    // if (mEnableOnlyExportCuttedWindow == false)
    // {
    //     SavePngDepthImage(pixels.data(), camera_width, camera_height,
    //                       path.c_str());
    // }
    // else
    // {
    //     std::vector<float> new_pixels(
    //         visible_window_width * visible_window_height, 0);
    //     for (int row = 0; row < visible_window_height; row++)
    //     {
    //         int new_pixels_st = visible_window_width * row;
    //         int old_pixels_st =
    //             (row + cast_range(1, 0)) * camera_width + cast_range(0, 0);
    //         memcpy(new_pixels.data() + new_pixels_st,
    //                pixels.data() + old_pixels_st,
    //                visible_window_width * sizeof(float));
    //     }

    // }
    // exit(0);
    // 4. download the result and save it to .ppm image
}

tMatrix4f get_discrete_mat(double vfov, int height, int width,
                           const tMatrix4f &view_mat_inv, bool is_vulkan)
{
    // printf("[debug] cursor xpos %.3f, ypos %.3f\n", xpos, ypos);

    tMatrix4f mat;
    // int height = mSwapChainExtent.height, width = mSwapChainExtent.width;
#ifdef __APPLE__
    xpos *= 2, ypos *= 2;
#endif
    // shape the conversion mat
    tMatrix4f mat1 = tMatrix4f::Identity();
    mat1(0, 0) = 1.0 / width;
    mat1(0, 3) = 0.5 / width;
    mat1(1, 1) = 1.0 / height;
    mat1(1, 3) = 0.5 / height;
    // std::cout << "after 1, vec = "
    //           << (test = mat1 * test).transpose() << std::endl;

    tMatrix4f mat2 = tMatrix4f::Identity();
    mat2(0, 0) = 2;
    mat2(0, 3) = -1;
    mat2(1, 1) = 2;
    mat2(1, 3) = -1;
    if (is_vulkan == true)
    {
        mat2.row(1) *= -1;
    }
    // std::cout << "after 2, vec = "
    //           << (test = mat2 * test).transpose() << std::endl;
    // mat3(0, 0) = std::tan(cMathUtil::Radians(mFov) / 2) * mNear;
    const float rad_fov = vfov / 180 * M_PI;
    float near_plane_dist = 1;
    // pos = mat2 * pos;
    tMatrix4f mat3 = tMatrix4f::Identity();
    mat3(0, 0) = width * 1.0 / height * std::tan(rad_fov / 2) * near_plane_dist;
    mat3(1, 1) = std::tan(rad_fov / 2) * near_plane_dist;
    mat3(2, 2) = 0, mat3(2, 3) = -near_plane_dist;
    // std::cout << "after 3, vec = "
    //           << (test = mat3 * test).transpose() << std::endl;

    // std::cout << "mat 3 = " << mat3 << std::endl;
    // exit(1);
    // pos = mat3 * pos;
    tMatrix4f mat4 = view_mat_inv;
    // std::cout << "after 4, vec = "
    //           << (test = mat4 * test).transpose() << std::endl;
    // std::cout <<"dir = " <<  (test -
    // mCamera->GetCameraPos()).normalized().transpose() << std::endl;
    mat = mat4 * mat3 * mat2 * mat1;
    return mat;
}

/**
 * \brief           Resize the info
 */
void cOptixRaycaster::resize(const tVector2i &newSize)
{
    // if window minimized
    if (newSize.x() == 0 || newSize.y() == 0)
        return;

    // resize our cuda frame buffer
    colorBuffer.resize(newSize.x() * newSize.y() * sizeof(uint32_t));

    // update the launch parameters that we'll pass to the optix
    // launch:
    launchParams.frame.size = newSize;
    launchParams.frame.colorBuffer = (uint32_t *)colorBuffer.d_pointer();

    // and re-set the camera, since aspect may have changed
    setCamera(lastSetCamera);
}

/**
 * \brief       set the camera info to the launch parameter (in the GPU)
 */
void cOptixRaycaster::setCamera(const CameraBasePtr &camera)
{
    lastSetCamera = camera;
    launchParams.position = camera->pos;

    // launchParams.convert_mat = tMatrix4f::Identity();
    tMatrix4f view_mat_inv =
        Eigen::lookAt(camera->pos, camera->center, camera->up).inverse();
    tMatrix4f tmp =
        get_discrete_mat(camera->fov, launchParams.frame.size[1],
                         launchParams.frame.size[0], view_mat_inv, true);
    // std::cout << "tmp = \n" << tmp << std::endl;
    // exit(0);
    // tVector3f y_axis = tVector3f(0, 1, 0);
    // y_axis = view_mat_inv.inverse().block(0, 0, 3, 3) * y_axis;
    // std::cout
    //     << "camera up = " << y_axis.transpose() << std::endl;
    // exit(0);
    launchParams.camera_up = camera->up;
    launchParams.camera_center = camera->center;
    launchParams.convert_mat = tmp;
}

void cOptixRaycaster::render()
{
    // sanity check: make sure we launch only after first resize is
    // already done:
    if (launchParams.frame.size.x() == 0)
        return;

    launchParamsBuffer.upload(&launchParams, 1);

    OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
                            pipeline, stream,
                            /*! parameters and SBT */
                            launchParamsBuffer.d_pointer(),
                            launchParamsBuffer.sizeInBytes, &sbt,
                            /*! dimensions of the launch: */
                            launchParams.frame.size.x(),
                            launchParams.frame.size.y(), 1));
    // sync - make sure the frame is rendered before we download and
    // display (obviously, for a high-performance application you
    // want to use streams and double-buffering, but for this simple
    // example, this will have to do)
    CUDA_SYNC_CHECK();
}

/**
 * \brief       download the color pixels (gray scale rgb channels)
 */

void cOptixRaycaster::downloadPixels(uint32_t h_pixels[])
{
    colorBuffer.download(h_pixels, launchParams.frame.size.x() *
                                       launchParams.frame.size.y());
}

void cOptixRaycaster::downloadPixels(float *h_pixels)
{
    colorBuffer.download(h_pixels, launchParams.frame.size.x() *
                                       launchParams.frame.size.y());
}

/**
 * \brief           calculate depth image for multiple camera views
 */
#include "utils/FileUtil.h"
#include "utils/SysUtil.h"
extern bool SaveEXRSingleChannel(const float *rgb, int width, int height,
                                 const char *outfilename);
void cOptixRaycaster::CalcDepthMapMultiCamera(
    const tMatrix2i &cast_range, int height, int width,
    std::vector<CameraBasePtr> camera_array,
    std::vector<std::string> path_array)
{
    // int st0 = cSysUtil::GetPhyMemConsumedBytes();
    CalcCastWindowSize(cast_range, visible_window_width, visible_window_height,
                       visible_window_st);
    launchParams.raycast_range = cast_range;
    Rebuild();
    // int st1 = cSysUtil::GetPhyMemConsumedBytes();

    this->camera_width = width;
    this->camera_height = height;

    int size = camera_array.size();
    SIM_ASSERT(size == path_array.size());
    for (int i = 0; i < size; i++)
    {
        if (true == cFileUtil::ExistsFile(path_array[i]))
        {

            // if (i == 0 && cMathUtil::RandInt(0, 100) < 1)
            {
                printf("[warn] depth img %s exist, ignore\n",
                       path_array[i].c_str());
            }
            continue;
        }
        // 1. if the size is changed, we need to resize the buffer
        setCamera(camera_array[i]);
        if (launchParams.frame.size.x() != width ||
            launchParams.frame.size.y() != height)
        {
            tVector2i size = tVector2i(width, height);
            resize(size);
        }
        // std::cout << "resize done\n";

        // UpdateVertexBufferToCuda();
        // 2. reupload all values in launch param
        BuildRandomSeries();
        render();
        // 3. launch the optix program
        // std::cout << "render done\n";
        // std::vector<uint32_t> pixels(width * height, 0);
        std::vector<float> pixels(width * height, 0);
        this->downloadPixels(pixels.data());
        // std::cout << "download pixels done\n";
        // save(path_array[i], width, height, pixels.data());
        std::string path = path_array[i];
        std::string suffix = cFileUtil::GetExtension(path);

        if (suffix == "exr")
        {
            SaveEXRSingleChannel(pixels.data(), width, height, path.c_str());
        }
        else if (suffix == "png")
        {
            SavePngDepthImage(pixels, path.c_str());
        }
        else
        {
            SIM_ERROR("Unsupported image type {}", suffix);
        }
        // SaveEXRDepthImage(const float *rgb, int width, int height, const
        // char*outfilename)
    }
    // int end = cSysUtil::GetPhyMemConsumedBytes();
    // std::cout << "CalcDepthMapMultiCamera add " << (end - st0) * 1e-6 << "
    // MB\n" ; std::cout << "CalcDepthMapMultiCamera0 add " << (st1 - st0) *
    // 1e-6 << " MB\n" ; std::cout << "CalcDepthMapMultiCamera1 add " << (end -
    // st1) * 1e-6 << " MB\n" ;
}

void cOptixRaycaster::CalcDepthMapMultiCamera(const tMatrix2i &cast_range,
                                              int height, int width,
                                              CameraBasePtr cur_cam,
                                              tVectorXf &pixels_eigen)
{
    launchParams.raycast_range = cast_range;
    Rebuild();

    this->camera_width = width;
    this->camera_height = height;

    // 1. if the size is changed, we need to resize the buffer
    setCamera(cur_cam);
    if (launchParams.frame.size.x() != width ||
        launchParams.frame.size.y() != height)
    {
        tVector2i size = tVector2i(width, height);
        resize(size);
    }
    // std::cout << "resize done\n";

    // UpdateVertexBufferToCuda();
    // 2. reupload all values in launch param
    render();
    // 3. launch the optix program
    // std::cout << "render done\n";
    // std::vector<uint32_t> pixels(width * height, 0);
    // std::vector<float> pixels(width * height, 0);
    downloadPixels(pixels_eigen.data());
    // std::cout << "download pixels done\n";
    // save(path_array[i], width, height, pixels.data());
    // SaveEXRDepthImage(pixels.data(), width, height, path_array[i].c_str());
    // SaveDepthEXR(const float *rgb, int width, int height, const
    // char*outfilename)
}

/**
 * \brief           Save png depth image
 */
#include "utils/TimeUtil.hpp"
bool cOptixRaycaster::SavePngDepthImage(const std::vector<float> &pixels,
                                        const char *path)
{
    // cTimeUtil::Begin("save_png");
    if (mEnableOnlyExportCuttedWindow == false)
    {
        SavePNGSingleChannel(pixels.data(), camera_width, camera_height, path);
    }
    else
    {

        std::vector<float> new_pixels(
            visible_window_width * visible_window_height, 0);

        for (int row = 0; row < visible_window_height; row++)
        {
            int new_pixels_st = visible_window_width * row;
            int old_pixels_st = (row + visible_window_st[1]) * camera_width +
                                visible_window_st[0];
            memcpy(new_pixels.data() + new_pixels_st,
                   pixels.data() + old_pixels_st,
                   visible_window_width * sizeof(float));
        }

        SavePNGSingleChannel(new_pixels.data(), visible_window_width,
                             visible_window_height, path);
    }

    // cTimeUtil::End("save_png");
    return true;
}

bool cOptixRaycaster::GetEnableOnlyExportingCuttedWindow() const
{
    return mEnableOnlyExportCuttedWindow;
}

/**
 * \brief           set the config for each object: enable depth value in
 * raycast or not...
 */
void cOptixRaycaster::InitEnableDepthForObjects(const Json::Value &conf)
{
    std::string preprocess_info_key = "preprocess_info";
    if (cJsonUtil::HasValue(preprocess_info_key, conf) == true)
    {
        Json::Value preinfo =
            cJsonUtil::ParseAsValue(preprocess_info_key, conf);
        Json::Value enable_depth_for_objs_json =
            cJsonUtil::ParseAsValue(DISABLE_DEPTH_FOR_OBJECTS_KEY, preinfo);
        // disable_depth_for_objects

        for (auto &key : enable_depth_for_objs_json.getMemberNames())
        {
            bool enable = enable_depth_for_objs_json[key].asBool();
            std::string name = key;
            disable_depth_for_objects[name] = enable;
        }
    }
}
#endif
