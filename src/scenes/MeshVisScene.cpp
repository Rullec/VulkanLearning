#include "MeshVisScene.h"
#include "utils/FileUtil.h"
#include "utils/JsonUtil.h"
#include <iostream>

#include "LinctexScene.h"
cMeshVisScene::cMeshVisScene()
{
    mCurMeshId = -1;
    mMeshDataDir = "";
    mMeshDataList.clear();
}

cMeshVisScene::~cMeshVisScene() {}
#include <algorithm>

bool comp_filename(const std::string &name0, const std::string &name1)
{
    int id0 =
        std::stoi(cFileUtil::RemoveExtension(cFileUtil::GetFilename(name0)));
    int id1 =
        std::stoi(cFileUtil::RemoveExtension(cFileUtil::GetFilename(name1)));
    return id0 < id1;
}
void cMeshVisScene::Init(const std::string &conf_path)
{
    Json::Value root;
    cJsonUtil::LoadJson(conf_path, root);
    cSimScene::Init(conf_path);
    InitGeometry(root);
    InitRaycaster();
    InitConstraint(root);
    InitDrawBuffer();

    mXcur = mClothInitPos;
    mMeshDataDir = cJsonUtil::ParseAsString(MESH_DATA_KEY, root);
    std::cout << "mesh data dir = " << mMeshDataDir << std::endl;
    SIM_ASSERT(cFileUtil::ExistsDir(mMeshDataDir));
    std::vector<std::string> files = cFileUtil::ListDir(mMeshDataDir);
    tVectorXd prop = tVectorXd::Zero(3);
    for (auto &x : files)
    {
        if (x.find(".json") != -1)
        {
            mMeshDataList.push_back(x);
            // std::cout << "add " << x << std::endl;
        }
    }

    SIM_ASSERT(mMeshDataList.size() > 0);

    // sort the mesh data by the index
    std::sort(mMeshDataList.begin(), mMeshDataList.end(), comp_filename);
    mCurMeshId = 0;
    SetMeshData(mCurMeshId);
}

void cMeshVisScene::Update(double dt)
{
    // std::cout << mXcur.segment(0, 10).transpose() << std::endl;
}

#ifdef _WIN32
#define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_INCLUDE_VULKAN
#include "GLFW/glfw3.h"
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#endif

#ifdef __linux__
#define VK_USE_PLATFORM_XCB_KHR
#include <GLFW/glfw3.h>
// #define GLFW_EXPOSE_NATIVE_
#include <GLFW/glfw3native.h>
#endif

#ifdef __APPLE__
#include <GLFW/glfw3.h>
#endif
void cMeshVisScene::Key(int key, int scanecode, int action, int mods)
{
    if (action != GLFW_PRESS)
    {
        return;
    }
    if (key == GLFW_KEY_LEFT || key == GLFW_KEY_UP)
    {
        mCurMeshId -= 1;
    }
    else if (key == GLFW_KEY_RIGHT || key == GLFW_KEY_DOWN)
    {
        mCurMeshId += 1;
    }
    int size = mMeshDataList.size();
    if (mCurMeshId >= size)
    {
        std::cout << (mCurMeshId > size) << std::endl;
        mCurMeshId = 0;
    }
    else if (mCurMeshId < 0)
    {
        mCurMeshId = mMeshDataList.size() - 1;
    }
    SetMeshData(mCurMeshId);
}

void cMeshVisScene::UpdateSubstep()
{
    std::cout << mXcur.segment(0, 10).transpose() << std::endl;
}

void cMeshVisScene::SetMeshData(int id)
{
    tVectorXd prop, pos;

    cLinctexScene::LoadSimulationData(pos, prop, mMeshDataList[id]);
    // std::cout << "pos = " << pos.segment(0, 10).transpose() << std::endl;
    UpdateCurNodalPosition(pos);
    UpdateRenderingResource();
    std::cout << "[log] cur mesh = " << mMeshDataList[id]
              << " feature = " << prop.transpose() << std::endl;
}