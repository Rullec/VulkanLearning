#include "ProcessTrainDataScene.h"
#include "cameras/ArcBallCamera.h"
#include "geometries/OptixRaycaster.h"
#include "geometries/Raycaster.h"
#include "geometries/Triangulator.h"
#include "utils/FileUtil.h"
#include "utils/JsonUtil.h"
#include "utils/TimeUtil.hpp"
#include <iostream>
cProcessTrainDataScene::cProcessTrainDataScene()
{
    mRaycaster = nullptr;
    mWidth = 0;
    mHeight = 0;
}
cProcessTrainDataScene::~cProcessTrainDataScene() {}
void cProcessTrainDataScene::Init(const std::string &conf_path)
{
    Json::Value root;
    cJsonUtil::LoadJson(conf_path, root);
    mGeometryInfoPath = cJsonUtil::ParseAsString(GEOMETRY_INFO_KEY, root);
    mRawDataDir = cJsonUtil::ParseAsString(RAW_DATA_DIR_KEY, root);
    mGenDataDir = cJsonUtil::ParseAsString(GEN_DATA_DIR_KEY, root);
    mWidth = cJsonUtil::ParseAsInt(DEPTH_IMAGE_WIDTH_KEY, root);
    mHeight = cJsonUtil::ParseAsInt(DEPTH_IMAGE_HEIGHT_KEY, root);
    mEnableClothGeometry =
        cJsonUtil::ParseAsBool(ENABLE_CLOTH_GEOMETRY_KEY, root);
    mCameraFov = cJsonUtil::ParseAsFloat(CAMERA_FOV_KEY, root);
    mExportImageType = cProcessTrainDataScene::GetImageType(
        cJsonUtil::ParseAsString(EXPORT_IMAGE_FORMAT_KEY, root));
    mCastingRange.row(0) =
        cJsonUtil::ReadVectorJson(
            cJsonUtil::ParseAsValue(CASTING_WIDTH_RANGE_KEY, root))
            .segment(0, 2)
            .transpose()
            .cast<int>();
    mCastingRange.row(1) =
        cJsonUtil::ReadVectorJson(
            cJsonUtil::ParseAsValue(CASTING_HEIGHT_RANGE_KEY, root))
            .segment(0, 2)
            .transpose()
            .cast<int>();
    mEnableOnlyExportingCuttedWindow =
        cJsonUtil::ParseAsBool(ENABLE_ONLY_EXPORTING_CUTTED_WINDOW_KEY, root);
    mCameraCenter = cJsonUtil::ReadVectorJson(
                        cJsonUtil::ParseAsValue(
                            cProcessTrainDataScene::CAMERA_CENTER_KEY, root))
                        .segment(0, 4);
    mCameraUp = cJsonUtil::ReadVectorJson(
                    cJsonUtil::ParseAsValue(
                        cProcessTrainDataScene::CAMERA_UP_KEY, root))
                    .segment(0, 4);
    mCameraPos = cJsonUtil::ReadVectorJson(
                     cJsonUtil::ParseAsValue(
                         cProcessTrainDataScene::CAMERA_POS_KEY, root))
                     .segment(0, 4);
    mImageUpsampling = cJsonUtil::ParseAsDouble(this->UPSAMPLING_KEY, root);
    SIM_ASSERT(mImageUpsampling > 0);
    // std::cout << "upsampling = " << mImageUpsampling << std::endl;
    mWidth = int(mWidth / mImageUpsampling);
    mHeight = int(mHeight / mImageUpsampling);
    mCastingRange =
        (mCastingRange.cast<float>() / mImageUpsampling).cast<int>();
    // std::cout << "width = " << mWidth << std::endl;
    // std::cout << "height = " << mHeight << std::endl;
    // std::cout << "cast range = \n" << mCastingRange << std::endl;
    // exit(0);

    InitPreprocessInfo(root["preprocess_info"]);
    InitObstacle(root);
    cTriangulator::LoadGeometry(mVertexArray, mEdgeArray, mTriangleArray,
                                mGeometryInfoPath);
    GenerateCameraViews();

    InitDrawBuffer();
    UpdateRenderingResource();
    ValidateOutputDir(mGenDataDir);
}

/**
 * \brief           Create and add obstacles
 */
void cProcessTrainDataScene::InitObstacle(const Json::Value &root)
{
    Json::Value sim_conf;
    cJsonUtil::LoadJson(cJsonUtil::ParseAsString(SIMULATION_CONF_KEY, root),
                        sim_conf);

    mEnableObstacle = cJsonUtil::ParseAsBool(ENABLE_OBSTACLE_KEY, sim_conf);
    if (mEnableObstacle == true)
    {
        CreateObstacle(
            cJsonUtil::ParseAsValue(cSimScene::OBSTACLE_CONF_KEY, sim_conf));
    }
}
/**
 * \brief       if the directory doesn't exist, create
 */
void cProcessTrainDataScene::ValidateOutputDir(std::string dir)
{
    if (cFileUtil::ExistsDir(dir) == false)
    {
        cFileUtil::CreateDir(dir.c_str());
    }
}
/**
 * \brief           Init the preprocessing info
 */
void cProcessTrainDataScene::InitPreprocessInfo(const Json::Value &conf)
{
    mPreprocessInfo.mNumOfInitRotationAngles =
        cJsonUtil::ParseAsInt(this->NUM_OF_INIT_ROTATION_ANGLE_KEY, conf);
    mPreprocessInfo.mNumOfClothRotationViews =
        cJsonUtil::ParseAsInt(this->NUM_OF_CLOTH_VIEWS_KEY, conf);
    mPreprocessInfo.mEnableCameraNoise =
        cJsonUtil::ParseAsBool(ENABLE_CAMERA_NOISE_KEY, conf);
    mPreprocessInfo.mCameraNoiseSamples =
        cJsonUtil::ParseAsInt(CAMERA_NOISE_SAMPLES_KEY, conf);

    mPreprocessInfo.mCameraTranslationNoise =
        cJsonUtil::ParseAsDouble(CAMERA_TRANSLATION_NOISE_KEY, conf);
    mPreprocessInfo.mCameraOrientationNoise =
        cJsonUtil::ParseAsDouble(CAMERA_ORIENTATION_NOISE_KEY, conf);
}
/**
 * \brief           Calc depth map (no cloth geometry)
 */
void cProcessTrainDataScene::CalcDepthMapNoCloth()
{
    printf("[warn] calculate depth map without cloth geometry\n");
    cTimeUtil::Begin("depth_without_cloth");
    auto ptr = std::dynamic_pointer_cast<cOptixRaycaster>(mRaycaster);
    std::vector<std::string> save_img_path_array(0);
    std::string suffix = this->GetImageSuffix(mExportImageType);
    for (int i = 0; i < mCameraLst.size(); i++)
    {
        // int i = 0;
        save_img_path_array.push_back(
            cFileUtil::ConcatFilename(mGenDataDir, std::to_string(i) + suffix));
    }
    // std::vector<CameraBasePtr> cam_views(0);
    // cam_views.push_back(mCamera);

    ptr->CalcDepthMapMultiCamera(mCastingRange, mHeight, mWidth, mCameraLst,
                                 save_img_path_array);
    printf("[log] calculate depth map without cloth geometry done, output dir "
           "= %s, cost %.3f ms\n",
           mGenDataDir.c_str(), cTimeUtil::End("depth_without_cloth", true));
}

/**
 * \brief           main loop to calculate the depth map for each data point
 */
#include "utils/SysUtil.h"
void cProcessTrainDataScene::CalcDepthMapLoop()
{
    std::vector<std::string> paths = cFileUtil::ListDir(this->mRawDataDir);
    SIM_ASSERT(paths.size() > 0);
    cTimeUtil::Begin("total");
    tVectorXd feature_vec_buf;

    // 1. for each mesh
    tVectorXd mesh_pos_vector;
    for (int mesh_id = 0; mesh_id < paths.size(); mesh_id++)
    {
        cTimeUtil::Begin("handle_img");
        std::string raycast_output_dir_level0 =
            mGenDataDir + "\\" + "mesh" + std::to_string(mesh_id);

        // 2. regenerate the camera pos & orientations
        GenerateCameraViews();

        // 3. for each init rotation angle, load the raw data
        std::string mesh_path = paths[mesh_id];
        tVectorXd mesh_feature_vec;
        SIM_ASSERT(true == LoadRawData(mesh_path, mesh_feature_vec));
        // the feature vector will be saved in the mesh directory
        mesh_pos_vector = mXcur;
        double rot_unit =
            (2 * M_PI / mPreprocessInfo.mNumOfClothRotationViews) /
            mPreprocessInfo.mNumOfInitRotationAngles;
        for (int init_rot_id = 0;
             init_rot_id < this->mPreprocessInfo.mNumOfInitRotationAngles;
             init_rot_id++)
        {
            // 4. restore the mesh
            mXcur.noalias() = mesh_pos_vector;
            std::string raycast_output_dir_level1 = raycast_output_dir_level0 +
                                                    "\\init_rot" +
                                                    std::to_string(init_rot_id);

            // 5. apply the init rotation angle

            // double rand_angle = cMathUtil::RandDouble(0, 2 * M_PI);
            tMatrix3d rotmat = cMathUtil::AxisAngleToRotmat(
                                   tVector(0, 1, 0, 0) * rot_unit * init_rot_id)
                                   .topLeftCorner<3, 3>();
            for (int i = 0; i < mXcur.size(); i += 3)
            {
                mXcur.segment(i, 3) = rotmat * mXcur.segment(i, 3);
            }
            UpdateCurNodalPosition(mXcur);

            // 5. given the camera list, given the num of cloth rotation views
            // printf("[debug] mesh %d init_rot_id %d output_dir: %s\n",
            // mesh_id,
            //        init_rot_id, raycast_output_dir_level1.c_str());

            CalcDepthMapMultiViews(raycast_output_dir_level1, this->mCameraLst,
                                   mPreprocessInfo.mNumOfClothRotationViews);
            // calculate & save the depth
        }

        // 6. save feature file
        {
            Json::Value feature_json;
            feature_json["feature"] =
                cJsonUtil::BuildVectorJson(mesh_feature_vec);
            cJsonUtil::WriteJson(cFileUtil::ConcatFilename(
                                     raycast_output_dir_level0, "feature.json"),
                                 feature_json, true);
        }
        printf("[log] handle mesh %d/%d, cost %.3f ms\n", mesh_id + 1,
               paths.size(), cTimeUtil::End("handle_img"));
    }
    cTimeUtil::End("total");
    // std::cout << "finished exit\n";
    // exit(1);
}
/**
 * \brief           Init raycaster
 */
#include "geometries/OptixRaycaster.h"
#include "sim/KinematicBody.h"
void cProcessTrainDataScene::InitRaycaster()
{
#ifdef USE_OPTIX
    mRaycaster =
        std::make_shared<cOptixRaycaster>(mEnableOnlyExportingCuttedWindow);
#else
    mRaycaster = std::make_shared<cRaycaster>(mEnableOnlyExportingCuttedWindow);
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
    std::cout << "[debug] add resources to raycaster done, num of obstacles = "
              << mObstacleList.size() << std::endl;
}
/**
 * \brief           calculate the depth image and do export
 */
void cProcessTrainDataScene::CalcDepthMap(const std::string raw_data_path,
                                          const std::string &save_png_path,
                                          const std::string &save_feature_path,
                                          CameraBasePtr camera)
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
    mRaycaster->CalcDepthMap(mCastingRange, mHeight, mWidth, camera,
                             save_png_path);
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
 * \param save_feature_path_array   save feature path array, precomputed as a
 * parameter \param camera_array              different camera views \param
 * num_of_rotation_view      different cloth rotation angle
 */
void cProcessTrainDataScene::CalcDepthMapMultiViews(
    const std::string &output_dir,
    const std::vector<CameraBasePtr> &camera_array, int num_of_rotation_view)
{
    if (cFileUtil::ExistsDir(output_dir) == false)
        cFileUtil::CreateDir(output_dir.c_str());
    // 2. calculate the transformation matrix which will be applied onto the
    // cloth surface
    tMatrix cloth_yaxis_rotmat =
        GenerateContinuousRotationMat(num_of_rotation_view);
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
            std::string output_dir_level1 =
                output_dir + "\\" + "cam" + std::to_string(j);
            if (cFileUtil::ExistsDir(output_dir_level1) == false)
                cFileUtil::CreateDir(output_dir_level1.c_str());
            png_array.push_back(cFileUtil::ConcatFilename(
                output_dir_level1, std::to_string(i) + ".png"));
            // printf("[debug] cam %d view %d path %s\n", j, i,
            //        png_array[png_array.size() - 1].c_str());
        }

        // 3.2 for loop over different camera view
        ptr->CalcDepthMapMultiCamera(mCastingRange, mHeight, mWidth,
                                     camera_array, png_array);
    }
}

/**
 * \brief       Generate rnadom camera views
 */
void cProcessTrainDataScene::GenerateCameraViews()
{
    mCameraLst.clear();
    const tVector3f &camera_pos = this->mCameraPos.segment(0, 3).cast<float>(),
                    &camera_center =
                        this->mCameraCenter.segment(0, 3).cast<float>(),
                    &camera_up = this->mCameraUp.segment(0, 3).cast<float>();

    auto cur_camera = std::make_shared<cArcBallCamera>(
        camera_pos, camera_center, camera_up, mCameraFov);

    mCameraLst.push_back(cur_camera);

    if (mPreprocessInfo.mEnableCameraNoise == true)
    {
        // std::cout << "begin to apply the camera noise\n";
        for (int i = 0; i < mPreprocessInfo.mCameraNoiseSamples; i++)
        {
            tVector3f cam_pos_noise = tVector3f::Random(3) *
                                      mPreprocessInfo.mCameraTranslationNoise,
                      cam_focus_noise = tVector3f::Random(3) *
                                        mPreprocessInfo.mCameraTranslationNoise,
                      cam_up_noise = tVector3f::Random(3) *
                                     mPreprocessInfo.mCameraOrientationNoise;
            // std::cout << "cam pos noise = " << cam_pos_noise.transpose() *
            // 1e2
            //           << " cm\n";
            // std::cout << "cam focus noise = "
            //           << cam_focus_noise.transpose() * 1e2 << " cm\n";
            // std::cout << "cam up noise = " << cam_up_noise.transpose() <<
            // "\n";
            auto new_camera = std::make_shared<cArcBallCamera>(
                camera_pos + cam_pos_noise, camera_center + cam_focus_noise,
                camera_up + cam_up_noise, mCameraFov);
            mCameraLst.push_back(new_camera);
        }
    }
    // exit(0);
}

void cProcessTrainDataScene::Update(double dt)
{
    if (mRaycaster == nullptr)
    {
        InitRaycaster();
    }

    if (mEnableClothGeometry == true)
    {
        CalcDepthMapLoop();
    }
    else
    {
        CalcDepthMapNoCloth();
    }
    exit(0);
}

void cProcessTrainDataScene::UpdateRenderingResource()
{
    cSimScene::UpdateRenderingResource();
}
void cProcessTrainDataScene::Reset() {}

void cProcessTrainDataScene::UpdateSubstep() {}

/**
 * \brief           Load raw data
 */
bool cProcessTrainDataScene::LoadRawData(std::string path,
                                         tVectorXd &feature_vec)
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
    return true;
}

/**
 * \brief           Calculate depth image here
 */
// extern int gWindowHeight, gWindowWidth;
extern tVector CalcCursorPointWorldPos_tool(double xpos, double ypos,
                                            int height, int width,
                                            const tMatrix &view_mat_inv);

void write_bmp(const std::string path, const int width, const int height,
               const int *const data)
{
    const int pad = (4 - (3 * width) % 4) % 4,
              filesize = 54 + (3 * width + pad) *
                                  height; // horizontal line must be a multiple
                                          // of 4 bytes long, header is 54 bytes
    char header[54] = {'B', 'M', 0, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0,  40,
                       0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0,  1, 0, 24, 0};
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
/**
 *
 */
tVectorXf cProcessTrainDataScene::CalcEmptyDepthImage(const tVector &cam_pos,
                                                      const tVector &cam_focus,
                                                      float fov)
{
    // 1. create cloth geometry
    mEnableClothGeometry = false;
    InitRaycaster();

    // 2. begin to create camera views
    mCameraPos = cam_pos;
    mCameraCenter = cam_focus;
    mCameraFov = fov;
    mCameraUp = tVector(0, 1, 0, 0);
    GenerateCameraViews();

    // 3. begin to calculate the depth image
    tVectorXf mPixels = tVectorXf::Zero(mHeight * mWidth);

    // 4.
    auto ptr = std::dynamic_pointer_cast<cOptixRaycaster>(mRaycaster);
    ptr->CalcDepthMapMultiCamera(mCastingRange, mHeight, mWidth, mCameraLst[0],
                                 mPixels);
    return mPixels;
}

/**
 * \brief           Get Depth image shape
 */
std::pair<int, int> cProcessTrainDataScene::GetDepthImageShape() const
{
    return std::pair<int, int>(mHeight, mWidth);
}

/**
 * \brief           Given image type string, return the type
 */
cProcessTrainDataScene::eImageType
cProcessTrainDataScene::GetImageType(const std::string name)
{
    for (int i = 0; i < eImageType::NUM_OF_TYPES; i++)
    {
        if (name == cProcessTrainDataScene::gImageSuffix[i])
        {
            return static_cast<eImageType>(i);
        }
    }

    SIM_ERROR("unrecognized image type");
    return eImageType::NUM_OF_TYPES;
}

/**
 * \brief           Given image type enum, return the suffix
 */
std::string cProcessTrainDataScene::GetImageSuffix(eImageType type)
{
    int id = static_cast<int>(type);
    SIM_ASSERT(id < eImageType::NUM_OF_TYPES);

    return cProcessTrainDataScene::gImageSuffix[id];
}
