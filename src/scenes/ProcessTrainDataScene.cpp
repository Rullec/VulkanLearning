#include "ProcessTrainDataScene.h"
#include "utils/JsonUtil.h"
#include "utils/FileUtil.h"
#include "utils/TimeUtil.hpp"
#include "geometries/Triangulator.h"
#include "cameras/ArcBallCamera.h"
#include "geometries/Raycaster.h"
#include "geometries/OptixRaycaster.h"
#include <iostream>
cProcessTrainDataScene::cProcessTrainDataScene()
{
    mWidth = 0;
    mHeight = 0;
}
cProcessTrainDataScene::~cProcessTrainDataScene()
{
}
void cProcessTrainDataScene::Init(const std::string &conf_path)
{
    Json::Value root;
    cJsonUtil::LoadJson(conf_path, root);
    mGeometryInfoPath = cJsonUtil::ParseAsString(GEOMETRY_INFO_KEY, root);
    mRawDataDir = cJsonUtil::ParseAsString(RAW_DATA_DIR_KEY, root);
    mGenDataDir = cJsonUtil::ParseAsString(GEN_DATA_DIR_KEY, root);
    mWidth = cJsonUtil::ParseAsInt(DEPTH_IMAGE_WIDTH_KEY, root);
    mHeight = cJsonUtil::ParseAsInt(DEPTH_IMAGE_HEIGHT_KEY, root);
    mEnableClothGeometry = cJsonUtil::ParseAsBool(ENABLE_CLOTH_GEOMETRY_KEY, root);
    // std::cout << "mEnableClothGeometry = " << mEnableClothGeometry << std::endl;
    // exit(0);
    // 1. validate the input/output dir
    // SIM_ASSERT(cFileUtil::ExistsDir(mRawDataDir) == true);
    if (cFileUtil::ExistsDir(mGenDataDir) == false)
    {
        cFileUtil::CreateDir(mGenDataDir.c_str());
        // cFileUtil::DeleteDir(mGenDataDir.c_str());
        // printf("[warn] the generated data dir %s is deleted\n", mGenDataDir.c_str());
    }
    {
        Json::Value sim_conf;
        cJsonUtil::LoadJson(cJsonUtil::ParseAsString(
                                SIMULATION_CONF_KEY, root),
                            sim_conf);

        mEnableObstacle = cJsonUtil::ParseAsBool(
            ENABLE_OBSTACLE_KEY, sim_conf);
        if (mEnableObstacle == true)
        {
            CreateObstacle(cJsonUtil::ParseAsValue(cSimScene::OBSTACLE_CONF_KEY, sim_conf));
        }
    }
    // 2. Init camera info
    InitCameraInfo(root);

    // 3. init geometry info
    cTriangulator::LoadGeometry(mVertexArray, mEdgeArray, mTriangleArray, mGeometryInfoPath);

    // 4. load a data info, set the vertex pos, init rendering resources
    InitCameraViews();
    InitRaycaster();
    if (mEnableClothGeometry == true)
    {
        CalcDepthMapLoop();
    }
    else
    {
        CalcDepthMapNoCloth();
    }
    exit(0);
    InitDrawBuffer();
    UpdateRenderingResource();
}

/**
 * \brief           Calc depth map (no cloth geometry)
*/
void cProcessTrainDataScene::CalcDepthMapNoCloth()
{
    printf("[warn] calculate depth map without cloth geometry\n");
    cTimeUtil::Begin("depth_without_cloth");
    auto ptr = std::dynamic_pointer_cast<cOptixRaycaster>(mRaycaster);
    std::vector<std::string> save_png_path_array(0);
    // for (int i = 0; i < mCameraViews.size(); i++)
    // {
    int i = 0;
    save_png_path_array.push_back(
        cFileUtil::ConcatFilename(mGenDataDir, std::to_string(i) + ".png"));
    // }
    std::vector<CameraBasePtr> cam_views(0);
    cam_views.push_back(mCamera);

    ptr->CalcDepthMapMultiCamera(
        mHeight, mWidth, cam_views, save_png_path_array);
    printf("[log] calculate depth map without cloth geometry done, output dir = %s, cost %.3f ms\n", mGenDataDir.c_str(), cTimeUtil::End("depth_without_cloth", true));
}

/**
 * \brief           main loop to calculate the depth map for each data point
*/
void cProcessTrainDataScene::CalcDepthMapLoop()
{
    std::vector<std::string> paths = cFileUtil::ListDir(this->mRawDataDir);
    SIM_ASSERT(paths.size() > 0);
    cTimeUtil::Begin("total");
    tVectorXd feature_vec_buf;

    // 1. for each mesh
    for (int i = 0; i < paths.size(); i++)
    {
        // 2. rotate the mesh, for each angle

        // 3. for each camera view (we may have the random noise)

        std::string surface_geo_data = paths[i];
        // std::vector<std::string> image_name_array(0);
        // std::vector<std::string> feature_name_array(0);
        int camera_id = 0;
        std::vector<CameraBasePtr> my_camera_views(0);
        my_camera_views.push_back(mCamera);
        // {
        //     std::string new_image_name = cFileUtil::RemoveExtension(cFileUtil::GetFilename(raw_data)) + "_" + std::to_string(camera_id) + ".png";
        //     std::string new_feature_name = cFileUtil::RemoveExtension(cFileUtil::GetFilename(raw_data)) + "_" + std::to_string(camera_id) + ".json";
        //     std::string new_full_image_name = cFileUtil::ConcatFilename(mGenDataDir, new_image_name);
        //     std::string new_full_feature_name = cFileUtil::ConcatFilename(mGenDataDir, new_feature_name);
        //     if (cFileUtil::ExistsFile(new_full_image_name) == true)
        //     {
        //         printf("[warn] file %s exist, ignore\n", new_image_name.c_str());
        //         my_camera_views.erase(my_camera_views.begin());
        //     }
        //     else
        //     {
        //         image_name_array.push_back(new_full_image_name);
        //         feature_name_array.push_back(new_full_feature_name);
        //     }
        // }

        // 2. create depth map
        cTimeUtil::Begin("handle_image");
        std::string base_name = cFileUtil::RemoveExtension(cFileUtil::GetFilename(surface_geo_data));
        CalcDepthMapMultiViews(surface_geo_data, base_name, my_camera_views, 10);
        printf("[log] handle image %d/%d, cost %.4f ms\n", i + 1, paths.size() + 1, cTimeUtil::End("handle_image", true));
    }
    cTimeUtil::End("total");
}
/**
 * \brief           Init raycaster
*/
#include "sim/KinematicBody.h"
#include "geometries/OptixRaycaster.h"
void cProcessTrainDataScene::InitRaycaster()
{
#ifdef USE_OPTIX
    mRaycaster = std::make_shared<cOptixRaycaster>();
#else
    mRaycaster = std::make_shared<cRaycaster>();
#endif
    if (mEnableClothGeometry == true)
    {
        mRaycaster->AddResources(mTriangleArray, mVertexArray);
    }
    else
    {
        printf("[warn] no cloth geometry in raycaster\n");
    }
    for (auto &x : mObstacleList)
    {
        auto obstacle_v_array = x->GetVertexArray();
        auto obstacle_triangle_array = x->GetTriangleArray();
        mRaycaster->AddResources(obstacle_triangle_array, obstacle_v_array);
    }
    std::cout << "[debug] add resources to raycaster done, num of obstacles = " << mObstacleList.size() << std::endl;
}
/**
 * \brief           calculate the depth image and do export
*/
void cProcessTrainDataScene::CalcDepthMap(const std::string raw_data_path, const std::string &save_png_path, const std::string &save_feature_path, CameraBasePtr camera)
{
    // 1. load data to vertex buffer
    tVectorXd feature_vec;
    // 2. create depth map
    if (false == LoadRawData(raw_data_path, feature_vec))
    {
        std::cout << "[warn] path " << raw_data_path << "invalid, ignore\n";
        return;
    }
    // std::cout << "feature vec = " << feature_vec.transpose() << std::endl;

    // InitRaycaster();

    // tMatrixXd res = CalcDepthImage();
    // std::cout << "begin to calc depth map\n";
    mRaycaster->CalcDepthMap(mHeight, mWidth, camera, save_png_path);
    {
        Json::Value value;
        value["feature"] = cJsonUtil::BuildVectorJson(feature_vec);
        cJsonUtil::WriteJson(save_feature_path, value);
    }
    // exit(0);
    // std::cout << "done\n";
    // std::cout << "[log] save png to " << save_png_path << std::endl;
}

tMatrix GenerateContinuousRotationMat(int divide_pairs)
{
    SIM_ASSERT(divide_pairs >= 1);
    if (divide_pairs == 1)
    {
        return tMatrix::Identity();
    }
    else
    {
        double angle = 2 * M_PI / divide_pairs;
        return cMathUtil::AxisAngleToRotmat(tVector(0, 1, 0, 0) * angle);
    }
}

/**
 * \brief                           Given feature vector
 * \param feature_vec               feature vector
 * \param save_png_path_array       save path array, precomputed as a parameter
 * \param save_feature_path_array   save feature path array, precomputed as a parameter
 * \param camera_array              different camera views
 * \param num_of_rotation_view      different cloth rotation angle
*/
void cProcessTrainDataScene::CalcDepthMapMultiViews(
    const std::string &surface_geo_path,
    const std::string &basename,
    const std::vector<CameraBasePtr> &camera_array,
    int num_of_rotation_view)
{
    // 1. load data from the path
    tVectorXd sim_param_vec;
    if (false == LoadRawData(surface_geo_path, sim_param_vec))
    {
        printf("[debug] Load geo info from %s failed, ignore\n", surface_geo_path.c_str());
        return;
    }

    // 2. calculate the transformation matrix which will be applied onto the cloth surface
    tMatrix cloth_yaxis_rotmat = GenerateContinuousRotationMat(num_of_rotation_view);
    auto ptr = std::dynamic_pointer_cast<cOptixRaycaster>(mRaycaster);
    std::vector<std::string> save_feature_path_array(0);
    // 3. for loop over different rotation angle
    for (int i = 0; i < num_of_rotation_view; i++)
    {
        // 3.1 apply current rotmat onto the vertex positions
        for (auto &x : mVertexArray)
        {
            x->mPos = cloth_yaxis_rotmat * x->mPos;
        }
        std::vector<std::string> png_array(0);
        for (int j = 0; j < camera_array.size(); j++)
        {
            auto new_base = basename + "_" + std::to_string(i) + "_" + std::to_string(j);
            png_array.push_back(cFileUtil::ConcatFilename(this->mGenDataDir, new_base + ".png"));
            save_feature_path_array.push_back(cFileUtil::ConcatFilename(this->mGenDataDir, new_base + ".json"));
        }

        // 3.2 for loop over different camera view
        ptr->CalcDepthMapMultiCamera(mHeight, mWidth, camera_array, png_array);
    }
    for (auto &tmp : save_feature_path_array)
    {
        if (true == cFileUtil::ExistsFile(tmp))
        {
            printf("[warn] feature file %s exist, ignore\n", tmp.c_str());
        }
        Json::Value value;
        value["feature"] = cJsonUtil::BuildVectorJson(sim_param_vec);
        cJsonUtil::WriteJson(tmp, value);
    }
}

/**
 * \brief       Generate rnadom camera views
*/
void cProcessTrainDataScene::InitCameraViews()
{
    const tVector3f &camera_pos = this->mCameraPos.segment(0, 3).cast<float>(),
                    &camera_center = this->mCameraCenter.segment(0, 3).cast<float>(),
                    &camera_up = this->mCameraUp.segment(0, 3).cast<float>();
    mCamera = std::make_shared<cArcBallCamera>(
        camera_pos,
        camera_center,
        camera_up);
}

void cProcessTrainDataScene::Update(double dt)
{
    UpdateRenderingResource();
}

void cProcessTrainDataScene::UpdateRenderingResource()
{
    cSimScene::UpdateRenderingResource();
}
void cProcessTrainDataScene::Reset()
{
}

/**
 * \brief           Init capture camera info
*/
void cProcessTrainDataScene::InitCameraInfo(const Json::Value &conf)
{
    mCameraCenter = cJsonUtil::ReadVectorJson(
                        cJsonUtil::ParseAsValue(cProcessTrainDataScene::CAMERA_CENTER_KEY, conf))
                        .segment(0, 4);
    mCameraUp = cJsonUtil::ReadVectorJson(
                    cJsonUtil::ParseAsValue(cProcessTrainDataScene::CAMERA_UP_KEY, conf))
                    .segment(0, 4);
    mCameraPos = cJsonUtil::ReadVectorJson(cJsonUtil::ParseAsValue(cProcessTrainDataScene::CAMERA_POS_KEY, conf)).segment(0, 4);
}

void cProcessTrainDataScene::UpdateSubstep()
{
}

/**
 * \brief           Load raw data
*/
bool cProcessTrainDataScene::LoadRawData(std::string path, tVectorXd &feature_vec)
{
    Json::Value root;
    cJsonUtil::LoadJson(path, root);
    tVectorXd input = cJsonUtil::ReadVectorJson(root["input"]);
    if (input.size() == 0)
        return false;
    feature_vec.noalias() = cJsonUtil::ReadVectorJson(root["output"]);
    // std::cout << "input size = " << input.size() << std::endl;
    SIM_ASSERT(input.size() == mVertexArray.size() * 3);
    UpdateCurNodalPosition(input);
}

/**
 * \brief           Calculate depth image here
*/
extern int gWindowHeight, gWindowWidth;
extern tVector CalcCursorPointWorldPos_tool(double xpos, double ypos, int height, int width, const tMatrix &view_mat_inv);

void write_bmp(const std::string path, const int width, const int height, const int *const data)
{
    const int pad = (4 - (3 * width) % 4) % 4, filesize = 54 + (3 * width + pad) * height; // horizontal line must be a multiple of 4 bytes long, header is 54 bytes
    char header[54] = {'B', 'M', 0, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 24, 0};
    for (int i = 0; i < 4; i++)
    {
        header[2 + i] = (char)((filesize >> (8 * i)) & 255);
        header[18 + i] = (char)((width >> (8 * i)) & 255);
        header[22 + i] = (char)((height >> (8 * i)) & 255);
    }
    char *img = new char[filesize];
    for (int i = 0; i < 54; i++)
        img[i] = header[i];
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            const int color = data[x + (height - 1 - y) * width];
            const int i = 54 + 3 * x + y * (3 * width + pad);
            img[i] = (char)(color & 255);
            img[i + 1] = (char)((color >> 8) & 255);
            img[i + 2] = (char)((color >> 16) & 255);
        }
        for (int p = 0; p < pad; p++)
            img[54 + (3 * width + p) + y * (3 * width + pad)] = 0;
    }
    std::ofstream file(path);
    file.write(img, filesize);
    file.close();
    delete[] img;
}

#include "omp.h"
tMatrixXd cProcessTrainDataScene::CalcDepthImageLegacy(const CameraBasePtr camera)
{
    tMatrixXd res = tMatrixXd::Zero(gWindowHeight, gWindowWidth);
    std::cout << "calc depth image, height " << gWindowHeight << " width " << gWindowWidth << std::endl;
    tVector camera_pos = cMathUtil::Expand(camera->pos.cast<double>(), 1);
    // tRay *ray = new tRay(camera_pos, camera_pos + tVector(0, 1, 0, 0));
    tVector intersection_pos;
    int max = -1;
#pragma omp parallel for
    for (int i_row = 0; i_row < gWindowHeight; i_row++)
    {
        std::cout << "processing row " << i_row << std::endl;
        for (int i_col = 0; i_col < gWindowHeight; i_col++)
        {
            tVector cursor_point = CalcCursorPointWorldPos_tool(i_row, i_col, gWindowHeight, gWindowWidth, camera->ViewMatrix().inverse().cast<double>());
            tRay *ray = new tRay(camera_pos, cursor_point);
            // ray->mDir = (cursor_point - camera_pos).normalized();

            tTriangle *tri = nullptr;
            int tri_id = -1;
            RayCastScene(ray, &tri, tri_id, intersection_pos);
            if (tri != nullptr)
            {
                // std::cout << "inter pos = " << intersection_pos.transpose() << std::endl;
                // std::cout << "camera pos = " << camera_pos.transpose() << std::endl;
                res(i_row, i_col) = (intersection_pos - camera_pos).norm();
                // max = std::max(max, res(i_row, i_col));
            }
            delete ray;
        }
    }

    // stbi_write
    //     stbi_write_png("test.png", gWindowWidth, gWindowHeight, 1, res.data(), sizeof(double));
    {
        double max_res = res.maxCoeff();
        std::cout << "max_res = " << max_res << std::endl;
        res /= max_res;
        Eigen::MatrixXi new_res = Eigen::MatrixXi::Zero(gWindowHeight, gWindowWidth);
        for (int i_row = 0; i_row < gWindowHeight; i_row++)
        {
            for (int i_col = 0; i_col < gWindowHeight; i_col++)
            {
                new_res(i_row, i_col) = int(255 * res(i_row, i_col));
            }
        }
        write_bmp("test.bmp", gWindowWidth, gWindowHeight, new_res.data());
    }
    std::cout << "done\n";
    // exit(0);
    return tMatrixXd::Zero(0, 0);
}