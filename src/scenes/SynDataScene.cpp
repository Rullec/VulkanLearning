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
    mSynDataNoise = std::make_shared<tSyncDataNoise>(
        cJsonUtil::ParseAsValue("noise", conf_json));
    mExportDataDir = cJsonUtil::ParseAsString(EXPORT_DATA_DIR, conf_json);
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
    Reset(); // wait for end, EngineStart = false

    if (cFileUtil::ExistsFile(GetExportFilename()) == true)
    {
        std::cout << "[warn] data point " << GetExportFilename()
                  << " exist, ignore\n";
        mTotalSamples_valid++;
        return;
    }
    mLinCloth->SetSimProperty(props);
    bool is_first_frame = true;
    buffer0.noalias() = tVectorXd::Zero(mLinCloth->GetClothFeatureSize());
    // std::cout << "buffer0 size = " << buffer0.size() << std::endl;
    double threshold = mConvergenceThreshold;
    int before_frame = this->mLinScene->GetCurrentFrame();
    int frame_check_gap = 100;
    mLinScene->Start();
    while (true)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        int cur_frame = mLinScene->GetCurrentFrame();
        // std::cout << "cur frame = " << cur_frame << std::endl;
        // begin to check the diff
        if ((cur_frame - before_frame) > frame_check_gap)
        {
            // fetch the cloth data
            // std::cout << "begin to check " << std::endl;
            mLinScene->Update(0);
            // std::cout << "begin to check 1 " << std::endl;
            buffer1.noalias() = mLinCloth->GetClothFeatureVector();
            int num_of_points = buffer1.size() / 3;
            // std::cout << "begin to check 2 " << std::endl;
            // std::cout << "buffer0 " << buffer0.size() << std::endl;
            // std::cout << "buffer1 " << buffer1.size() << std::endl;
            tVectorXd diff_vec = buffer1 - buffer0;
            // std::cout << "begin to check 2.5 " << std::endl;
            Eigen::Map<
                Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>>
                diff_mat(diff_vec.data(), num_of_points, 3);
            // std::cout << "begin to check 3 " << std::endl;
            tVectorXd rowwise_norm = diff_mat.rowwise().norm();
            // std::cout << "begin to check 4 " << std::endl;
            double max_move_dist = rowwise_norm.maxCoeff();
            // std::cout << "max move dist = " << max_move_dist << std::endl;
            if (max_move_dist < threshold)
            {
                printf("[debug] %d RunSimulation: cur frame %d, diff norm %.6f "
                       "< %.6f, converge for feature ",
                       mTotalSamples_count, cur_frame, max_move_dist,
                       threshold);
                std::cout << props->BuildVisibleFeatureVector().transpose()
                          << std::endl;
                break;
            }
            buffer0.noalias() = buffer1;
        }
    }
    // export data
    if (mEnableDataCleaner == false)
    {
        // old behavior
        // 1. form the export data path (along with the directory)
        std::string full_name = GetExportFilename();
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
            std::string full_name = this->GetExportFilename();
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

/**
 * \brief           Get export filename
 */
std::string cSynDataScene::GetExportFilename() const
{
    std::string single_name = std::to_string(mTotalSamples_valid) + ".json";
    std::string full_name =
        cFileUtil::ConcatFilename(mExportDataDir, single_name);
    return full_name;
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
    num_of_samples *= this->mSynDataNoise->mNumOfSamplesPerProp;
    printf("[log] we will have %d samples\n", num_of_samples);
    for (int i = 0; i < num_of_properties; i++)
    {
        // 1. reset the internal linctex scene,
        mLinScene->Reset();
        // 2. get the simulation parameter

        auto prop = mPropManager->GetProperty(i);

        for (int i = 0; i < mSynDataNoise->mNumOfSamplesPerProp; i++)
        {
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
    ApplyNoiseIfPossible();
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
    mExportDataDir = "data/export_data/" + mExportDataDir;
    // mExportDataDir = str_replace(
    //     str_replace("data/export_data/" + cTimeUtil::GetSystemTime(), " ",
    //     "_"),
    //     ":", "_");

    std::cout << "[debug] export_data dir = " << mExportDataDir << std::endl;

    // if (cFileUtil::ExistsDir(mExportDataDir) == true)
    // {
    //     SIM_ERROR("{} exist, exit", mExportDataDir);
    //     exit(0);
    // }
    if (cFileUtil::ExistsDir(mExportDataDir) == false)
    {
        cFileUtil::CreateDir(mExportDataDir.c_str());
    }
    // SIM_ASSERT(cFileUtil::CreateDir(mExportDataDir.c_str()) == true);
    SIM_ASSERT(cFileUtil::ExistsDir(mExportDataDir) == true);
}

/**
 * \brief       init the data augmentation strucutre
 */
#define _USE_MATH_DEFINES
#include <math.h>
cSynDataScene::tSyncDataNoise::tSyncDataNoise(const Json::Value &conf)
{
    mEnableNoise = cJsonUtil::ParseAsBool("enable_noise", conf);
    mNumOfSamplesPerProp = cJsonUtil::ParseAsInt("samples_per_prop", conf);
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
    // std::cout << mNumOfSamplesPerProp << " " << mEnableInitYRotation << " "
    // << mEnableInitYPosNoise << " " << this->mInitYPosNoiseStd << std::endl;
    // exit(0);
}

/**
 * \brief       apply the radom y rotation and y noise in linctex scene
 */
void cSynDataScene::ApplyNoiseIfPossible()
{
    if (mSynDataNoise->mEnableNoise == true)
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
    else
    {
        std::cout << "[debug] noise is disabled\n";
    }
}
#endif // _WIN32
