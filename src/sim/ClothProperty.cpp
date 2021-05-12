#ifdef _WIN32
#include "ClothProperty.h"
#include "utils/JsonUtil.h"
#include "utils/LogUtil.h"
#include <iostream>
void tPhyProperty::Init(const Json::Value &root)
{
    Json::Value conf = cJsonUtil::ParseAsValue("cloth_property", root);
    std::cout << "[debug] init physical prop = \n " << conf << std::endl;
    mStretchWarp = cJsonUtil::ParseAsDouble(mPropertiesName[0], conf);
    mStretchWeft = cJsonUtil::ParseAsDouble(mPropertiesName[1], conf);
    mBendingWarp = cJsonUtil::ParseAsDouble(mPropertiesName[2], conf);
    mBendingWeft = cJsonUtil::ParseAsDouble(mPropertiesName[3], conf);
    mBendingBias = cJsonUtil::ParseAsDouble(mPropertiesName[4], conf);
    SIM_ASSERT(conf.size() == mNumOfProperties);
}
tVectorXd tPhyProperty::BuildFullFeatureVector() const
{
    tVectorXd feature = tVectorXd::Zero(mNumOfProperties);
    feature[0] = mStretchWarp;
    feature[1] = mStretchWeft;
    feature[2] = mBendingWarp;
    feature[3] = mBendingWeft;
    feature[4] = mBendingBias;
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
    else if (name == "bending_warp")
    {
        return 2;
    }
    else if (name == "bending_weft")
    {
        return 3;
    }
    else if (name == "bending_bias")
    {
        return 4;
    }
    else
    {
        SIM_ERROR("unrecognized feature name {}", name);
        exit(0);
    }
    return -1;
}
/**
 * \brief           Given a full feature vector, load its value from vector to discrete values
*/
void tPhyProperty::ReadFeatureVector(const tVectorXd &vec)
{
    SIM_ASSERT(mNumOfProperties == vec.size());
    mStretchWarp = vec[0];
    mStretchWeft = vec[1];
    mBendingWarp = vec[2];
    mBendingWeft = vec[3];
    mBendingBias = vec[4];
}

/**
 * \brief           Given a json value and a bool vector, init the property
*/
void tBatchProperty::SetVisilibities(std::vector<bool> visibilities, const tVectorXi &visible_faeture_index)
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
#endif