#ifdef _WIN32
#include "SynDataScene.h"
#include "LinctexScene.h"
#include "scenes/LinctexCloth.h"
#include "sim/cloth/ClothProperty.h"
#include "sim/cloth/ClothPropertyManager.h"
#include "utils/FileUtil.h"
#include "utils/JsonUtil.h"
#include "utils/TimeUtil.hpp"
#include <iostream>
// int online_cur_prop_id = -1;
// tVectorXd online_before_nodal_pos;

extern void DumpSimulationData(const tVectorXd &simualtion_result,
                               const tVectorXd &simulation_property,
                               // const tVector &init_rot_qua,
                               // const tVector &init_translation,
                               const std::string &filename);
extern void LoadSimulationData(tVectorXd &simualtion_result,
                               tVectorXd &simulation_property,
                               const std::string &filename);
cSynDataScene::cSynDataScene()
{
    mSynDataNoise = nullptr;
    mLinScene = nullptr;
    mDefaultConfigPath = "";
    cMathUtil::SeedRand(0);
}

std::string str_replace(std::string full_raw_str, std::string from,
                        std::string to)
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
    mDefaultConfigPath =
        cJsonUtil::ParseAsString("default_config_path", conf_json);
    mEnableDataAug = cJsonUtil::ParseAsBool("enable_noise", conf_json);
    mConvergenceThreshold =
        cJsonUtil::ParseAsDouble("convergence_threshold", conf_json);
    mEnableDataCleaner =
        cJsonUtil::ParseAsBool("enable_data_cleaner", conf_json);
    mDataCleanerThreshold =
        cJsonUtil::ParseAsDouble("data_cleaner_threshold", conf_json);
    mPropManager = std::make_shared<tPhyPropertyManager>(
        cJsonUtil::ParseAsValue("property_manager", conf_json));
    mEnableDraw = cJsonUtil::ParseAsBool(this->ENABLE_DRAW_KEY, conf_json);
    // std::cout << "enable noise = " << mEnableDataAug << std::endl;
    if (mEnableDataAug == true)
    {
        mSynDataNoise = std::make_shared<tSyncDataNoise>(
            cJsonUtil::ParseAsValue("noise", conf_json));
    }
    mLinScene = std::make_shared<cLinctexScene>();
    mLinScene->Init(mDefaultConfigPath);
    mLinCloth = mLinScene->GetLinctexCloth();
    InitExportDataDir();

    if (mEnableDraw == false)
    {
        OfflineSampling();
    }
    else
    {
    }
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
 * \brief           Given a simulatin property, run the simulation and get the
 * result
 */

int mTotalSamples_valid = 0;
int mTotalSamples_count = 0;
void cSynDataScene::RunSimulation(tPhyPropertyPtr props)
{
    // std::cout << "run sim for feature = " < < < < std::endl;
    // std::cout << "feature size = " << props->BuildFullFeatureVector().size()
    // << std::endl;
    mTotalSamples_count++;
    Reset();

    // mLinScene->ApplyTransform(init_trans);
    mLinCloth->SetSimProperty(props);
    bool is_first_frame = true;
    buffer0.noalias() = tVectorXd::Zero(mLinCloth->GetClothFeatureSize());
    double threshold = mConvergenceThreshold;
    int cur_iters = 0;
    int min_iters = 5;
    while (++cur_iters)
    {
        mLinScene->Update(1e-3);
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        buffer1.noalias() = mLinCloth->GetClothFeatureVector();

        // find the biggest movement vertex
        // {
        int num_of_points = buffer1.size() / 3;
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>>
            buffer0_mat(buffer0.data(), num_of_points, 3);
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>>
            buffer1_mat(buffer1.data(), num_of_points, 3);

        double max_move_dist = (buffer0 - buffer1).rowwise().norm().maxCoeff();
        // }
        // printf("%d size %d size \n", buffer0.size(), buffer1.size());
        // double diff_norm = (buffer1 - buffer0).norm();
        double diff_norm = max_move_dist;
        // printf("[debug] before norm %.6f, cur norm %.6f, diff %.6f\n",
        // buffer0.norm(), buffer1.norm(), diff_norm);
        if ((diff_norm < threshold && is_first_frame == false &&
             cur_iters > min_iters))
        {
            printf("[debug] %d RunSimulation: iters %d, diff norm %.6f < %.6f, "
                   "converge for feature ",
                   mTotalSamples_count, cur_iters, diff_norm, threshold);
            std::cout << props->BuildVisibleFeatureVector().transpose()
                      << std::endl;
            break;
        }
        if (is_first_frame == true && diff_norm > threshold)
        {
            is_first_frame = false;
        }
        buffer0 = buffer1;
    }
    // export data
    if (mEnableDataCleaner == false)
    {
        // old behavior
        // 1. form the export data path (along with the directory)
        std::string single_name = std::to_string(mTotalSamples_valid) + ".json";
        std::string full_name =
            cFileUtil::ConcatFilename(mExportDataDir, single_name);
        // 2. "input" & output
        DumpSimulationData(
            mLinCloth->GetClothFeatureVector(),
            props->BuildVisibleFeatureVector(),
            // cMathUtil::QuaternionToCoef(cMathUtil::RotMatToQuaternion(init_trans)),
            // init_trans.block(0, 3, 4, 1),
            full_name);
        mTotalSamples_valid++;
    }
    else
    {
        if (false == CheckDuplicateWithDataSet())
        {
            std::string single_name =
                std::to_string(mTotalSamples_valid) + ".json";
            std::string full_name =
                cFileUtil::ConcatFilename(mExportDataDir, single_name);
            // 2. "input" & output
            DumpSimulationData(
                mLinCloth->GetClothFeatureVector(),
                props->BuildVisibleFeatureVector(),
                // cMathUtil::QuaternionToCoef(cMathUtil::RotMatToQuaternion(init_trans)),
                // init_trans.block(0, 3, 4, 1),
                full_name);
            mTotalSamples_valid++;
        }
    }
}

double calc_dist(const tVectorXd &v0, const tVectorXd &v1)
{
    double cur_dist = -1;
    SIM_ASSERT(v0.size() == v1.size());
    for (int i = 0; i < v1.size(); i += 3)
    {
        double dist = (v0.segment(i, 3) - v1.segment(i, 3)).norm();
        if (dist > cur_dist)
            cur_dist = dist;
    }
    return cur_dist;
}
/**
 * \brief
 */
bool cSynDataScene::CheckDuplicateWithDataSet() const
{
    tVectorXd old_res, old_prop;
    tVectorXd cur_res = mLinCloth->GetClothFeatureVector();
    for (int i = mTotalSamples_valid - 1; i >= 0; i--)
    {
        std::string single_name = std::to_string(i) + ".json";
        std::string full_name =
            cFileUtil::ConcatFilename(mExportDataDir, single_name);
        LoadSimulationData(old_res, old_prop, full_name);

        // 1. calc distance
        double cur_dist = calc_dist(old_res, cur_res);
        if (cur_dist < mDataCleanerThreshold)
        {
            printf("[debug] compared with %s, the dist = %.4f < %.4f, "
                   "duplicate!\n",
                   single_name.c_str(), cur_dist, mDataCleanerThreshold);
            return true;
        }
    }
    return false;
}
/**
 * \brief       ultimate run method
 */
void cSynDataScene::OfflineSampling()
{
    // 1. fetch all settings
    int total_sample = 0;
    int num_of_properties = mPropManager->GetNumOfProperties();
    int num_of_samples = num_of_properties;
    if (mEnableDataAug == true)
        num_of_samples *= this->mSynDataNoise->mNumOfNoisedSamples;
    printf("[log] we will have %d samples\n", num_of_samples);
    for (int i = 0; i < num_of_properties; i++)
    {
        // 1. reset the internal linctex scene,
        mLinScene->Reset();
        // 2. get the simulation parameter

        auto prop = mPropManager->GetProperty(i);

        // std::cout << "full feature = " <<
        // prop->BuildFullFeatureVector().transpose() << std::endl; std::cout <<
        // "output feature = " << prop->BuildVisibleFeatureVector().transpose()
        // << std::endl; continue;
        if (mEnableDataAug == true)
        {
            // SIM_ERROR("hasn't finished yet ");
            // exit(0);
            // std::cout << "num of noised samples = " <<
            // mSynDataNoise->mNumOfNoisedSamples << std::endl; exit(0);
            for (int i = 0; i < mSynDataNoise->mNumOfNoisedSamples; i++)
            {
                cTimeUtil::Begin("run_sim");
                RunSimulation(prop);
                cTimeUtil::End("run_sim");
                total_sample++;
            }
        }
        else
        {
            // 3. get the <simulation result - parameter> to json
            cTimeUtil::Begin("run_sim");
            RunSimulation(prop);
            cTimeUtil::End("run_sim");
            total_sample++;
        }
    }

    printf("[log] total samples = %d, real unduplicate samples = %d\n",
           total_sample, mTotalSamples_valid);
    exit(0);
}
void cSynDataScene::Update(double dt)
{
    cSimScene::Update(dt);

    // if (online_cur_prop_id != -1)
    {
        // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        mLinScene->Update(dt);
        mLinScene->UpdateRenderingResource();
    }
}
void cSynDataScene::Reset()
{
    mLinScene->Reset();
    if (mEnableDataAug == true)
    {
        ApplyNoiseIfPossible();
    }
}
const tVectorXf &cSynDataScene::GetTriangleDrawBuffer()
{
    // std::cout << "[debug] syn_scene : get triangles size = " << tmp.size() <<
    // std::endl;
    return mLinScene->GetTriangleDrawBuffer();
}
const tVectorXf &cSynDataScene::GetEdgesDrawBuffer()
{
    // auto tmp = ;
    // std::cout << "[debug] syn_scene : get edges size = " << tmp.size() <<
    // std::endl;
    return mLinScene->GetEdgesDrawBuffer();
}

bool cSynDataScene::CreatePerturb(tRay *ray)
{
    return mLinScene->CreatePerturb(ray);
}

void cSynDataScene::CursorMove(int xpos, int ypos)
{
    mLinScene->CursorMove(xpos, ypos);
}

void cSynDataScene::Key(int key, int scancode, int action, int mods)
{
    mLinScene->Key(key, scancode, action, mods);
}
void cSynDataScene::MouseButton(int button, int action, int mods)
{
    mLinScene->MouseButton(button, action, mods);
}

/**
 * \brief               Create the export data directory
 */
void cSynDataScene::InitExportDataDir()
{
    mExportDataDir = str_replace(
        str_replace("data/export_data/" + cTimeUtil::GetSystemTime(), " ", "_"),
        ":", "_");

    std::cout << "[debug] target export data = " << mExportDataDir << std::endl;

    if (cFileUtil::ExistsDir(mExportDataDir) == true)
    {
        SIM_ERROR("{} exist, exit", mExportDataDir);
        exit(0);
    }
    SIM_ASSERT(cFileUtil::CreateDir(mExportDataDir.c_str()) == true);
    SIM_ASSERT(cFileUtil::ExistsDir(mExportDataDir) == true);
}

/**
 * \brief       init the data augmentation strucutre
 */
#define _USE_MATH_DEFINES
#include <math.h>
cSynDataScene::tSyncDataNoise::tSyncDataNoise(const Json::Value &conf)
{
    mNumOfNoisedSamples = cJsonUtil::ParseAsInt("noised_samples", conf);
    // mEnableInitYRotation = cJsonUtil::ParseAsBool("enable_init_rotation",
    // conf); mEnableFoldNoise = cJsonUtil::ParseAsBool("enable_fold_noise",
    // conf); mEnableInitYPosNoise =
    //     cJsonUtil::ParseAsBool("enable_gaussian_pos_noise", conf);
    // mInitYPosNoiseStd = cJsonUtil::ParseAsDouble("gaussian_std", conf);
    // mFoldCoef = cJsonUtil::ParseAsDouble("fold_coef", conf);
    mEnableLowFreqNoise =
        cJsonUtil::ParseAsDouble("enable_low_freq_noise", conf);
    mMaxFoldAmp = cJsonUtil::ParseAsDouble("max_fold_amp", conf);
    mMinFoldNum = cJsonUtil::ParseAsInt("min_fold_num", conf);
    mMaxFoldNum = cJsonUtil::ParseAsInt("max_fold_num", conf);
    // SIM_ASSERT(mEnableInitYRotation == false);
    // SIM_ASSERT(mEnableFoldNoise == true);
    // std::cout << mNumOfNoisedSamples << " " << mEnableInitYRotation << " " <<
    // mEnableInitYPosNoise << " " << this->mInitYPosNoiseStd << std::endl;
    // exit(0);
}

/**
 * \brief       apply the radom y rotation and y noise in linctex scene
 */
void cSynDataScene::ApplyNoiseIfPossible()
{
    if (mEnableDataAug)
    {
        // double theta = 0;
        // mLinScene->ApplyNoise(this->mSynDataNoise->mEnableInitYRotation,
        // theta, mSynDataNoise->mEnableInitYPosNoise,
        // mSynDataNoise->mInitYPosNoiseStd);
        // if (this->mSynDataNoise->mEnableFoldNoise == true)
        // {

        //     tVector3d principle_axis = tVector3d::Random();
        //     principle_axis[1] = 0;
        //     principle_axis.normalize();

        //     mLinScene->ApplyFoldNoise(principle_axis,
        //     mSynDataNoise->mFoldCoef); std::cout << "[debug] apply fold noise
        //     along principle noise "
        //               << principle_axis.transpose()
        //               << ", folding coef = " << mSynDataNoise->mFoldCoef
        //               << std::endl;
        // }

        // // if(this-)
        // if (mSynDataNoise->mEnableInitYPosNoise == true)
        // {
        //     double angle = 0;
        //     mLinScene->ApplyNoise(mSynDataNoise->mEnableInitYRotation, angle,
        //                           mSynDataNoise->mEnableInitYPosNoise,
        //                           mSynDataNoise->mInitYPosNoiseStd);
        //     std::cout << "[debug] apply gaussian noise on Y axis, std = "
        //               << mSynDataNoise->mInitYPosNoiseStd << std::endl;
        // }

        if (mSynDataNoise->mEnableLowFreqNoise == true)
        {
            int num = cMathUtil::RandInt(mSynDataNoise->mMinFoldNum,
                                         mSynDataNoise->mMaxFoldNum);
            mLinCloth->ApplyMultiFoldsNoise(num, mSynDataNoise->mMaxFoldAmp);
            std::cout << "[debug] apply low freq noise " << num << std::endl;
        }
        // std::cout << "theta = " << theta << std::endl;
        // std::cout << "std = " << mSynDataNoise->mInitYPosNoiseStd <<
        // std::endl; exit(0);
    }
}
#endif // _WIN32
