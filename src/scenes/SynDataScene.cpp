#ifdef _WIN32
#include "SynDataScene.h"
#include "utils/JsonUtil.h"
#include "utils/TimeUtil.hpp"
#include "LinctexScene.h"
#include "sim/ClothPropertyManager.h"
#include "sim/ClothProperty.h"
#include <iostream>
#include "utils/FileUtil.h"
cSynDataScene::cSynDataScene()
{
    mLinScene = nullptr;
    mDefaultConfigPath = "";
}

std::string str_replace(std::string full_raw_str, std::string from, std::string to)
{
    while (full_raw_str.find(from) != std::string::npos)
    {
        int st_pos = full_raw_str.find(from);
        full_raw_str.replace(st_pos, from.length(), to);
    }

    return full_raw_str;
}
void cSynDataScene::Init(const std::string &conf_path)
{
    Json::Value conf_json;
    cJsonUtil::LoadJson(conf_path, conf_json);
    mDefaultConfigPath = cJsonUtil::ParseAsString("default_config_path", conf_json);
    mPropManager = std::make_shared<tPhyPropertyManager>(cJsonUtil::ParseAsValue("property_manager", conf_json));

    mLinScene = std::make_shared<cLinctexScene>();
    mLinScene->Init(mDefaultConfigPath);
    InitExportDataDir();
}

std::string to_string(const tVectorXd &vec)
{
    std::string str = "";
    for (int i = 0; i < vec.size(); i++)
    {
        str += std::to_string(vec[i]);
        if (i != (vec.size() - 1))
            str += "_";
    }
    return str;
}
/**
 * \brief           Given a simulatin property, run the simulation and get the result 
*/
int mTotalSamples = 0;
void cSynDataScene::RunSimulation(tPhyPropertyPtr props)
{
    // std::cout << "run sim for feature = " < < < < std::endl;
    // std::cout << "feature size = " << props->BuildFeatureVector().size() << std::endl;
    Reset();
    mLinScene->SetSimProperty(props);
    bool is_first_frame = true;
    buffer0.noalias() = tVectorXd::Zero(mLinScene->GetClothFeatureSize());
    double threshold = 0.5;
    int cur_iters = 0;
    int min_iters = 5;
    while (++cur_iters)
    {
        mLinScene->Update(1e-3);
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        buffer1.noalias() = mLinScene->GetClothFeatureVector();
        // printf("%d size %d size \n", buffer0.size(), buffer1.size());
        double diff_norm = (buffer1 - buffer0).norm();
        // printf("[debug] before norm %.6f, cur norm %.6f, diff %.6f\n", buffer0.norm(), buffer1.norm(), diff_norm);
        if (
            (diff_norm < threshold && is_first_frame == false && cur_iters > min_iters))
        {
            printf("[debug] RunSimulation: iters %d, diff norm %.6f < %.6f, converge for feature ", cur_iters, diff_norm, threshold);
            std::cout << props->BuildFeatureVector().transpose() << std::endl;
            break;
        }
        if (is_first_frame == true && diff_norm > threshold)
        {
            is_first_frame = false;
        }
        buffer0 = buffer1;
    }
    mTotalSamples++;
    // export data
    {
        // 1. form the export data path (along with the directory)
        std::string single_name = std::to_string(mTotalSamples) + ".json";
        std::string full_name = cFileUtil::ConcatFilename(mExportDataDir, single_name);
        // 2. "input"
        Json::Value export_json;
        export_json["input"] = cJsonUtil::BuildVectorJson(mLinScene->GetClothFeatureVector());
        export_json["output"] = cJsonUtil::BuildVectorJson(props->BuildFeatureVector());
        std::cout << "feature = " << props->BuildFeatureVector().transpose() << std::endl;
        cJsonUtil::WriteJson(full_name, export_json);
        std::cout << "[debug] save data to " << single_name << std::endl;
        // 3. "output"
    }
}
/**
 * \brief       ultimate run method
*/
void cSynDataScene::Update(double dt)
{
    // 1. fetch all settings
    int total_sample = 0;
    while (true)
    {
        // 1. reset the internal linctex scene,
        mLinScene->Reset();
        // 2. get the simulation parameter
        if (mPropManager->IsEnd() == true)
        {
            printf("total sampleds %d, exit\n", total_sample);
            break;
        }
        auto prop = mPropManager->GetNextProperty();
        // 3. get the <simulation result - parameter> to json
        cTimeUtil::Begin("run_sim");
        RunSimulation(prop);
        cTimeUtil::End("run_sim");
        total_sample++;
    }
    exit(0);
}
void cSynDataScene::Reset()
{
    mLinScene->Reset();
}
const tVectorXf &cSynDataScene::GetTriangleDrawBuffer()
{
    // std::cout << "[debug] syn_scene : get triangles size = " << tmp.size() << std::endl;
    return mLinScene->GetTriangleDrawBuffer();
}
const tVectorXf &cSynDataScene::GetEdgesDrawBuffer()
{
    // auto tmp = ;
    // std::cout << "[debug] syn_scene : get edges size = " << tmp.size() << std::endl;
    return mLinScene->GetEdgesDrawBuffer();
}

bool cSynDataScene::CreatePerturb(tRay *ray)
{
    return mLinScene->CreatePerturb(ray);
}

void cSynDataScene::CursorMove(cDrawScene *draw_scene, int xpos, int ypos)
{
    mLinScene->CursorMove(draw_scene, xpos, ypos);
}
void cSynDataScene::MouseButton(cDrawScene *draw_scene, int button, int action,
                                int mods)
{
    mLinScene->MouseButton(
        draw_scene, button, action,
        mods);
}

void cSynDataScene::UpdateSubstep()
{
}

/**
 * \brief               Create the export data directory
*/
void cSynDataScene::InitExportDataDir()
{
    mExportDataDir =
        str_replace(
            str_replace(
                "export_data/" + cTimeUtil::GetSystemTime(),
                " ",
                "_"),
            ":", "_");

    std::cout << "[debug] target export data = " << mExportDataDir << std::endl;

    if (cFileUtil::ExistsDir(mExportDataDir) == true)
    {
        SIM_ERROR("{} exist, exit", mExportDataDir);
        exit(0);
    }
    cFileUtil::CreateDir(mExportDataDir.c_str());
    SIM_ASSERT(cFileUtil::ExistsDir(mExportDataDir) == true);
}
#endif _WIN32
