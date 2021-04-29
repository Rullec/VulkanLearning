#include "ProcessTrainDataScene.h"
#include "utils/JsonUtil.h"
#include "utils/FileUtil.h"
#include "utils/TimeUtil.hpp"
#include "geometries/Triangulator.h"
#include "cameras/ArcBallCamera.h"
#include "geometries/Raycaster.h"
#include <iostream>
cProcessTrainDataScene::cProcessTrainDataScene()
{
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

    // 1. validate the input/output dir
    // SIM_ASSERT(cFileUtil::ExistsDir(mRawDataDir) == true);
    if (cFileUtil::ExistsDir(mGenDataDir) == false)
    {
        cFileUtil::CreateDir(mGenDataDir.c_str());
        // cFileUtil::DeleteDir(mGenDataDir.c_str());
        // printf("[warn] the generated data dir %s is deleted\n", mGenDataDir.c_str());
    }

    // 2. Init camera info
    InitCameraInfo(root);

    // 3. init geometry info
    cTriangulator::LoadGeometry(mVertexArray, mEdgeArray, mTriangleArray, mGeometryInfoPath);

    // 4. load a data info, set the vertex pos, init rendering resources
    InitCameraViews();
    InitRaycaster();
    std::vector<std::string> paths = cFileUtil::ListDir(this->mRawDataDir);
    SIM_ASSERT(paths.size() > 0);
    cTimeUtil::Begin("process");
    int total_samples = 0;
    for (int i = 0; i < paths.size(); i++)
    {
        std::string raw_data = paths[i];
        for (int camera_id = 0; camera_id < this->mCameraViews.size(); camera_id++)
        {
            std::string new_image_name = cFileUtil::RemoveExtension(cFileUtil::GetFilename(raw_data)) + "_" + std::to_string(camera_id) + ".png";
            std::string new_feature_name = cFileUtil::RemoveExtension(cFileUtil::GetFilename(raw_data)) + "_" + std::to_string(camera_id) + ".json";
            std::string new_full_image_name = cFileUtil::ConcatFilename(mGenDataDir, new_image_name);
            std::string new_full_feature_name = cFileUtil::ConcatFilename(mGenDataDir, new_feature_name);
            if(cFileUtil::ExistsFile(new_full_image_name) == true)
            {
                printf("[warn] file %s exist, ignore\n", new_image_name.c_str());
                continue;
            }
            CalcDepthMap(raw_data, new_full_image_name, new_full_feature_name, mCameraViews[camera_id]);
            printf("[log] raw data %s saved to %s\n", cFileUtil::GetFilename(raw_data).c_str(), new_image_name.c_str());
            total_samples++;
            // exit(0);
        }
        // if (total_samples > 10)
        //     break;
    }
    cTimeUtil::End("process");
    std::cout << "[log] total samples = " << total_samples << std::endl;
    // {
    //     // CalcDepthMap(raw_data, "tmp1.png", mCameraViews[1]);
    //     // CalcDepthMap(raw_data, "tmp2.png", mCameraViews[2]);
    //     // CalcDepthMap(raw_data, "tmp3.png", mCameraViews[3]);
    // }
    exit(0);
    InitDrawBuffer();
    UpdateRenderingResource();
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
    int height = 800, width = 800;
    // std::cout << "begin to calc depth map\n";
    mRaycaster->CalcDepthMap(height, width, camera, save_png_path);
    {
        Json::Value value;
        value["feature"] = cJsonUtil::BuildVectorJson(feature_vec);
        cJsonUtil::WriteJson(save_feature_path, value);
    }
    // exit(0);
    // std::cout << "done\n";
    // std::cout << "[log] save png to " << save_png_path << std::endl;
}

void cProcessTrainDataScene::InitCameraViews()
{
    mCameraViews.resize(mCameraPos.size(), nullptr);
    for (int i = 0; i < mCameraPos.size(); i++)
    {
        const tVector3f &camera_pos = this->mCameraPos[i].segment(0, 3).cast<float>(),
                        &camera_center = this->mCameraCenter.segment(0, 3).cast<float>(),
                        &camera_up = this->mCameraUp.segment(0, 3).cast<float>();
        mCameraViews[i] = std::make_shared<cArcBallCamera>(
            camera_pos,
            camera_center,
            camera_up);
    }
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

    Json::Value pos_lst = cJsonUtil::ParseAsValue(cProcessTrainDataScene::CAMERA_POS_KEY, conf);
    SIM_ASSERT(pos_lst.size() >= 1);
    mCameraPos.clear();
    for (int i = 0; i < pos_lst.size(); i++)
    {
        tVector tmp = cJsonUtil::ReadVectorJson(pos_lst[i]).segment(0, 4);
        // std::cout << "pos " << i << " = " << tmp.transpose() << std::endl;
        mCameraPos.push_back(tmp);
    }
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