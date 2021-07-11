#ifdef _WIN32
#include "ClothProperty.h"
#include "utils/JsonUtil.h"
#include "utils/LogUtil.h"
#include <iostream>
void tPhyProperty::Init(const Json::Value &root)
{
    Json::Value conf = cJsonUtil::ParseAsValue("cloth_property", root);
    // std::cout << "[debug] init physical prop = \n " << conf << std::endl;
    mStretchWarp = cJsonUtil::ParseAsDouble(mPropertiesName[0], conf);
    mStretchWeft = cJsonUtil::ParseAsDouble(mPropertiesName[1], conf);
    mStretchBias = cJsonUtil::ParseAsDouble(mPropertiesName[2], conf);
    mBendingWarp = cJsonUtil::ParseAsDouble(mPropertiesName[3], conf);
    mBendingWeft = cJsonUtil::ParseAsDouble(mPropertiesName[4], conf);
    mBendingBias = cJsonUtil::ParseAsDouble(mPropertiesName[5], conf);

    // begin to test
    // {
    //     for (float gui_value = 0; gui_value <= 100; gui_value+=0.5)
    //     {
    //         float sim_value = ConvertBendingCoefFromGUIToSim(gui_value);
    //         float gui_value_re = ConvertBendingCoefFromSimToGUI(sim_value);
    //         // printf("[bending] gui value %.7f, sim value %.7f, gui value re
    //         %.7f\n",
    //         //        gui_value, sim_value, gui_value_re);
    //         SIM_ASSERT(std::fabs(gui_value_re - gui_value) < 1e-4);
    //         sim_value = ConvertStretchCoefFromGUIToSim(gui_value);
    //         gui_value_re = ConvertStretchCoefFromSimToGUI(sim_value);
    //         // printf("[stretch] gui value %.7f, sim value %.7f, gui value re
    //         %.7f\n",
    //         //        gui_value, sim_value, gui_value_re);
    //         SIM_ASSERT(std::fabs(gui_value_re - gui_value) < 1e-4);
    //     }
    //     // std::cout << "test succ\n";
    //     // exit(0);
    // }
    SIM_ASSERT(conf.size() == mNumOfProperties);
}
tVectorXd tPhyProperty::BuildFullFeatureVector() const
{
    tVectorXd feature = tVectorXd::Zero(mNumOfProperties);
    feature[0] = mStretchWarp;
    feature[1] = mStretchWeft;
    feature[2] = mStretchBias;
    feature[3] = mBendingWarp;
    feature[4] = mBendingWeft;
    feature[5] = mBendingBias;
    return feature;
}

/**
 * \brief           Given a name, return the feature value
 */
double tPhyProperty::GetFeature(std::string name) const
{
    if (name == "stretch_warp")
    {
        return mStretchWarp;
    }
    else if (name == "stretch_weft")
    {
        return mStretchWeft;
    }
    else if (name == "stretch_bias")
    {
        return mStretchBias;
    }
    else if (name == "bending_warp")
    {
        return mBendingWarp;
    }
    else if (name == "bending_weft")
    {
        return mBendingWeft;
    }
    else if (name == "bending_bias")
    {
        return mBendingBias;
    }
    else
    {
        SIM_ERROR("unrecognized feature name {}", name);
        exit(0);
    }
    return 0;
}

/**
 * \brief           Given a name, get feature index
 */
int tPhyProperty::GetFeatureIdx(std::string name)
{
    if (name == "stretch_warp")
    {
        return 0;
    }
    else if (name == "stretch_weft")
    {
        return 1;
    }
    else if (name == "stretch_bias")
    {
        return 2;
    }
    else if (name == "bending_warp")
    {
        return 3;
    }
    else if (name == "bending_weft")
    {
        return 4;
    }
    else if (name == "bending_bias")
    {
        return 5;
    }
    else
    {
        SIM_ERROR("unrecognized feature name {}", name);
        exit(0);
    }
    return -1;
}
/**
 * \brief           Given a full feature vector, load its value from vector to
 * discrete values
 */
void tPhyProperty::ReadFeatureVector(const tVectorXd &vec)
{
    SIM_ASSERT(mNumOfProperties == vec.size());
    mStretchWarp = vec[0];
    mStretchWeft = vec[1];
    mStretchBias = vec[2];
    mBendingWarp = vec[3];
    mBendingWeft = vec[4];
    mBendingBias = vec[5];
}

/**
 * \brief           Given a json value and a bool vector, init the property
 */
void tBatchProperty::SetVisilibities(std::vector<bool> visibilities,
                                     const tVectorXi &visible_faeture_index)
{
    SIM_ASSERT(visibilities.size() == mNumOfProperties);
    for (int i = 0; i < mNumOfProperties; i++)
    {
        mVisibility[i] = visibilities[i];
    }

    mVisibleFeatureIndex = visible_faeture_index;
    SIM_ASSERT(mVisibleFeatureIndex.size() == mNumOfProperties);
    mNumOfVisibleFeature = 0;
    for (int i = 0; i < mNumOfProperties; i++)
    {
        if (mVisibleFeatureIndex[i] != -1)
        {
            mNumOfVisibleFeature++;
        }
    }

    // verify times
    tVectorXi times = tVectorXi::Zero(mNumOfVisibleFeature);
    for (int i = 0; i < mNumOfProperties; i++)
    {
        if (mVisibleFeatureIndex[i] != -1)
        {
            times[mVisibleFeatureIndex[i]] += 1;
        }
    }

    for (int i = 0; i < mNumOfVisibleFeature; i++)
    {
        SIM_ASSERT(times[i] == 1);
    }
}

/**
 * \brief           Build visible feature vector
 */
tVectorXd tPhyProperty::BuildVisibleFeatureVector() const
{
    SIM_ASSERT(false && "unsupported vec");
    exit(0);
    return tVectorXd::Zero(0);
}
tVectorXd tBatchProperty::BuildVisibleFeatureVector() const
{
    tVectorXd visible_feature = tVectorXd::Zero(mNumOfVisibleFeature);
    for (int i = 0; i < mNumOfProperties; i++)
    {
        int vis_id = mVisibleFeatureIndex[i];
        if (vis_id != -1)
        {
            visible_feature[vis_id] = GetFeature(mPropertiesName[i]);
        }
    }
    return visible_feature;
}

std::vector<std::pair<float, float>>
    stretch_map_guitosim = {{0.0f, 1e3f},  {9.0f, 1e4f},  {27.0f, 1e5f},
                            {57.0f, 4e5f}, {93.0f, 4e6f}, {100.0f, 1e7f}},
    bending_map_guitosim = {{0.0f, 0e0f},  {10.0f, 1e2f}, {28.0f, 1e3f},
                            {48.0f, 3e3f}, {65.0f, 2e4f}, {83.0f, 2e5f},
                            {100.0f, 2e6f}};

float CalculateSimValueFromGUI(
    const std::vector<std::pair<float, float>> &arrays, float gui_value)
{
    // 1. exceed the boundary
    int last_id = arrays.size() - 1;
    float threshold = 1e-6;
    if (gui_value >= arrays[last_id].first - threshold)
    {
        return arrays[last_id].second;
    }
    if (gui_value <= arrays[0].first + threshold)
    {
        return arrays[0].second;
    }

    // 2. find the interval
    for (int st = 0; st < arrays.size() - 1; st++)
    {
        float st_key = arrays[st].first;
        float ed_key = arrays[st + 1].first;
        float st_value = arrays[st].second;
        float ed_value = arrays[st + 1].second;
        float gap = ed_key - st_key;
        if ((st_key <= gui_value) && (ed_key >= gui_value))
        {
            // do interplotion and return
            return (1.0 - (gui_value - st_key) / gap) * st_value +
                   (1.0 - (ed_key - gui_value) / gap) * ed_value;
        }
    }
    return std::nan("");
}

float CalculateGUIValueFromSim(
    const std::vector<std::pair<float, float>> &arrays, float sim_value)
{
    // 1. exceed the boundary
    int last_id = arrays.size() - 1;
    float threshold = 1e-6;
    if (sim_value < arrays[0].second)
    {
        return arrays[0].first;
    }
    if (arrays[last_id].second < sim_value)
    {
        return arrays[last_id].first;
    }

    // 2. find the interval
    for (int st = 0; st < arrays.size() - 1; st++)
    {
        float st_key = arrays[st].second, ed_key = arrays[st + 1].second,
              st_value = arrays[st].first, ed_value = arrays[st + 1].first;
        float gap = ed_key - st_key;
        if ((st_key <= sim_value) && (ed_key >= sim_value))
        {
            return (1.0 - (sim_value - st_key) / gap) * st_value +
                   (1.0 - (ed_key - sim_value) / gap) * ed_value;
        }
    }
    return std::nan("");
}
/**
 * \brief           convert gui bending coef to simulation coef
 */
float tPhyProperty::ConvertBendingCoefFromGUIToSim(float gui_value)
{
    // 1. judge illegal
    return CalculateSimValueFromGUI(bending_map_guitosim, gui_value);
}
/**
 * \brief           convert simulation bending coef to gui coef
 */
float tPhyProperty::ConvertBendingCoefFromSimToGUI(float sim_value)
{
    return CalculateGUIValueFromSim(bending_map_guitosim, sim_value);
}
/**
 * \brief           convert stretch gui coef to simulation coef
 */
float tPhyProperty::ConvertStretchCoefFromGUIToSim(float gui_value)
{
    return CalculateSimValueFromGUI(stretch_map_guitosim, gui_value);
}
/**
 * \brief           convert stretch simulation coef to gui coef
 */
float tPhyProperty::ConvertStretchCoefFromSimToGUI(float sim_value)
{
    return CalculateGUIValueFromSim(stretch_map_guitosim, sim_value);
}
#endif
