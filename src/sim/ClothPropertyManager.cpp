#ifdef _WIN32
#include "ClothPropertyManager.h"
#include "ClothProperty.h"
#include "utils/JsonUtil.h"
#include "utils/LogUtil.h"
#include <iostream>

tPhyPropertyManager::tPhyPropertyManager(
    const Json::Value &conf)
{
    const std::string mode = cJsonUtil::ParseAsString("mode", conf);
    if (mode == "log")
    {
        this->mSampleMode = eSampleMode::LOG;
    }
    else if (mode == "linear")
    {
        this->mSampleMode = eSampleMode::LINEAR;
    }
    else
    {
        SIM_ERROR("UNSUPPORTED PROPERTY SAMPLE MODE {}", mode);
    }
    const Json::Value range_json = cJsonUtil::ParseAsValue("prop_range", conf);

    InitPropRange(range_json);

    mSamples = cJsonUtil::ParseAsInt("sample", conf);
    SIM_ASSERT(mSamples > 0);
    mNextSampleIndices.resize(tPhyProperty::mNumOfProperties, 0);
    mFirstSample = true;
    std::cout << "sample " << mSamples << std::endl;
}

tPhyPropertyPtr tPhyPropertyManager::GetNextProperty()
{
    if (mFirstSample == true)
    {
        mFirstSample = false;
    }
    {
        // if (this->IsEnd() == true)
        // {
        //     SIM_ERROR("now the property is end, failed to get next property");
        // }
        // else
        // {
        // }
    }
    auto ptr = std::make_shared<tPhyProperty>();
    ptr->ReadFeatureVector(CalcPropertyFromIndices(mNextSampleIndices));
    AddIndices();
    return ptr;
}

bool tPhyPropertyManager::IsEnd() const
{
    if (mFirstSample)
    {
        return false;
    }
    else
    {
        bool is_end = true;
        for (int i = 0; i < tPhyProperty::mNumOfProperties; i++)
        {
            SIM_ASSERT((mNextSampleIndices[i] >= 0) &&
                       (mNextSampleIndices[i] < mSamples));
            if (mNextSampleIndices[i] != 0)
            {
                is_end = false;
                break;
            }
        }
        return is_end;
    }
}

void tPhyPropertyManager::InitPropRange(const Json::Value &prop_range_json)
{
    mPropMin = tVectorXd::Zero(tPhyProperty::mNumOfProperties);
    mPropMax = tVectorXd::Zero(tPhyProperty::mNumOfProperties);
    mPropDefault = tVectorXd::Zero(tPhyProperty::mNumOfProperties);
    for (int i = 0; i < tPhyProperty::mNumOfProperties; i++)
    {
        tVector3d feature_range = cJsonUtil::ReadVectorJson(
                                      cJsonUtil::ParseAsValue(tPhyProperty::mPropertiesName[i], prop_range_json))
                                      .segment(0, 3);
        mPropMin[i] = feature_range[0];
        mPropDefault[i] = feature_range[1];
        mPropMax[i] = feature_range[2];
    }
    SIM_ASSERT(prop_range_json.size() == tPhyProperty::mNumOfProperties);
    std::cout << "[debug] raw prop min = " << mPropMin.transpose() << std::endl;
    std::cout << "[debug] raw prop max = " << mPropMax.transpose() << std::endl;
    std::cout << "[debug] raw prop default = " << mPropDefault.transpose() << std::endl;
    // exit(0);
}

void tPhyPropertyManager::AddIndices()
{
    mNextSampleIndices[tPhyProperty::mNumOfProperties - 1]++;
    for (int idx = tPhyProperty::mNumOfProperties - 1; idx >= 0; idx--)
    {
        // division
        if (idx >= 1)
        {
            mNextSampleIndices[idx - 1] += std::floor(mNextSampleIndices[idx] / mSamples);
        }
        mNextSampleIndices[idx] %= mSamples;
    }
}

/**
 * \brief           
*/
#include <cmath>
tVectorXd tPhyPropertyManager::CalcPropertyFromIndices(const std::vector<int> &indices) const
{
    tVectorXd prop = tVectorXd::Zero(tPhyProperty::mNumOfProperties);
    for (int i = 0; i < tPhyProperty::mNumOfProperties; i++)
    {
        int id = indices[i];
        int gap = mSamples - 1;
        double min_log_value = std::log(mPropMin[i]);
        double incremental =
            (gap > 0 ? (std::log(mPropMax[i]) - std::log(mPropMin[i])) / (gap) : 0) * id;
        prop[i] = std::exp(min_log_value + incremental);
    }
    return prop;
}

void tPhyPropertyManager::PrintIndices()
{
    std::cout << "[debug] cur indices = ";
    for (auto &x : mNextSampleIndices)
        std::cout << x;
    std::cout << std::endl;
}
#endif