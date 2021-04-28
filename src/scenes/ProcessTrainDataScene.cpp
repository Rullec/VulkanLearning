#include "ProcessTrainDataScene.h"
#include "utils/JsonUtil.h"
#include "utils/FileUtil.h"
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
    SIM_ASSERT(cFileUtil::ExistsDir(mRawDataDir) == true);
    if (cFileUtil::ExistsDir(mGenDataDir))
    {
        cFileUtil::DeleteDir(mGenDataDir.c_str());
        printf("[warn] the generated data dir %s is deleted\n", mGenDataDir.c_str());
    }
    cFileUtil::CreateDir(mGenDataDir.c_str());

    // 2. Init camera info
    InitCameraInfo(root);

    // 3. init geometry info
    cTriangulator::LoadGeometry(mVertexArray, mEdgeArray, mTriangleArray, mGeometryInfoPath);
    LoadRawData();
    InitDrawBuffer();
    UpdateRenderingResource();

    // 4. load a data info, set the vertex pos, init rendering resources
    InitRaycaster();

    for (int i = 0; i < 1; i++)
    {
        const tVector3f &camera_pos = this->mCameraPos[0].segment(0, 3).cast<float>(),
                        &camera_center = this->mCameraCenter.segment(0, 3).cast<float>(),
                        &camera_up = this->mCameraUp.segment(0, 3).cast<float>();
        mCamera = std::make_shared<cArcBallCamera>(
            camera_pos,
            camera_center,
            camera_up);
        // tMatrixXd res = CalcDepthImage();
        int height = 800, width = 800;
        std::cout << "begin to calc depth map\n";
        mRaycaster->CalcDepthMap(height, width, mCamera);
        std::cout << "done\n";
        // exit(0);
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
void cProcessTrainDataScene::LoadRawData()
{
    std::vector<std::string> paths = cFileUtil::ListDir(this->mRawDataDir);
    SIM_ASSERT(paths.size() > 0);
    std::string path = paths[0];
    Json::Value root;
    cJsonUtil::LoadJson(path, root);
    tVectorXd input = cJsonUtil::ReadVectorJson(root["input"]);
    std::cout << "input size = " << input.size() << std::endl;
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
tMatrixXd cProcessTrainDataScene::CalcDepthImage()
{
    tMatrixXd res = tMatrixXd::Zero(gWindowHeight, gWindowWidth);
    std::cout << "calc depth image, height " << gWindowHeight << " width " << gWindowWidth << std::endl;
    tVector camera_pos = cMathUtil::Expand(this->mCamera->pos.cast<double>(), 1);
    // tRay *ray = new tRay(camera_pos, camera_pos + tVector(0, 1, 0, 0));
    tVector intersection_pos;
    int max = -1;
#pragma omp parallel for
    for (int i_row = 0; i_row < gWindowHeight; i_row++)
    {
        std::cout << "processing row " << i_row << std::endl;
        for (int i_col = 0; i_col < gWindowHeight; i_col++)
        {
            tVector cursor_point = CalcCursorPointWorldPos_tool(i_row, i_col, gWindowHeight, gWindowWidth, mCamera->ViewMatrix().inverse().cast<double>());
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