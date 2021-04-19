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
    SIM_ASSERT(conf.size() == mNumOfProperties);
}
tVectorXd tPhyProperty::BuildFeatureVector() const
{
    tVectorXd feature = tVectorXd::Zero(mNumOfProperties);
    feature[0] = mStretchWarp;
    feature[1] = mStretchWeft;
    feature[2] = mBendingWarp;
    feature[3] = mBendingWeft;
    return feature;
}

void tPhyProperty::ReadFeatureVector(const tVectorXd &vec)
{
    SIM_ASSERT(mNumOfProperties == vec.size());
    mStretchWarp = vec[0];
    mStretchWeft = vec[1];
    mBendingWarp = vec[2];
    mBendingWeft = vec[3];
}

#endif