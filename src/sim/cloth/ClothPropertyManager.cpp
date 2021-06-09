#ifdef _WIN32
#include "ClothPropertyManager.h"
#include "ClothProperty.h"
#include "utils/JsonUtil.h"
// #include "utils/LogUtil.h"
#include <iostream>

tPhyPropertyManager::tPhyPropertyManager(const Json::Value &conf)
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

    // 1. load the value range and their visibilities
    {
        Json::Value props = cJsonUtil::ParseAsValue("properties", conf);
        SIM_ASSERT(props.size() == tPhyProperty::mNumOfProperties);
        mVisibilities.resize(tPhyProperty::mNumOfProperties);

        mPropMin = tVectorXd::Zero(tPhyProperty::mNumOfProperties);
        mPropMax = tVectorXd::Zero(tPhyProperty::mNumOfProperties);
        mSamples = tVectorXi::Zero(tPhyProperty::mNumOfProperties);
        mVisibleIndex = tVectorXi::Ones(tPhyProperty::mNumOfProperties) * -1;
        for (int i = 0; i < tPhyProperty::mNumOfProperties; i++)
        {
            Json::Value sub_value = cJsonUtil::ParseAsValue(
                tPhyProperty::mPropertiesName[i], props);
            // std::cout << sub_value << std::endl;
            tVectorXd range = cJsonUtil::ReadVectorJson(sub_value["range"]);
            mSamples[i] = cJsonUtil::ParseAsInt("num_of_samples", sub_value);
            mPropMin[i] = range[0];
            mPropMax[i] = range[1];
            mVisibilities[i] = cJsonUtil::ParseAsBool("visible", sub_value);
            if (mVisibilities[i] == true)
            {
                mVisibleIndex[i] = cJsonUtil::ParseAsInt("index", sub_value);
            }
            std::cout << "[debug] property " << tPhyProperty::mPropertiesName[i]
                      << " range = " << range.transpose()
                      << " samples = " << mSamples[i]
                      << " visi = " << mVisibilities[i] << std::endl;
        }
        std::cout << "visible index = " << mVisibleIndex.transpose()
                  << std::endl;
        InitExchangeablePairs(
            cJsonUtil::ParseAsValue("exchangeable_pairs", conf));
        // exit(0);
        InitFeatures();
    }

    // for (int i = 0; i < this->GetNumOfProperties(); i++)
    // {
    //     auto prop = GetProperty(i);
    //     printf("-------------%d----------\n", i);
    //     std::cout << "full feature = " <<
    //     prop->BuildFullFeatureVector().transpose() << std::endl; std::cout <<
    //     "visible feature = " << prop->BuildVisibleFeatureVector().transpose()
    //     << std::endl;
    // }
    // exit(0);
    // std::cout << "sample " << mSamplePerProperty << std::endl;
}

/**
 * \brief               Init exchangeable pairs
 */
void tPhyPropertyManager::InitExchangeablePairs(const Json::Value &conf)
{
    for (int i = 0; i < conf.size(); i++)
    {
        std::string name0 = conf[i][0].asString(),
                    name1 = conf[i][1].asString();
        int idx0 = tPhyProperty::GetFeatureIdx(name0),
            idx1 = tPhyProperty::GetFeatureIdx(name1);
        mExchangeablePairs.push_back(std::pair<int, int>(idx0, idx1));
        auto a = mExchangeablePairs[mExchangeablePairs.size() - 1];
        printf("[debug] add exchangeable pair %d-%d\n", a.first, a.second);
    }
}

tPhyPropertyPtr tPhyPropertyManager::GetProperty(int i)
{
    // if (mFirstSample == true)
    // {
    //     mFirstSample = false;
    // }
    // {
    //     // if (this->IsEnd() == true)
    //     // {
    //     //     SIM_ERROR("now the property is end, failed to get next
    //     property");
    //     // }
    //     // else
    //     // {
    //     // }
    // }
    auto ptr = std::make_shared<tBatchProperty>();
    ptr->ReadFeatureVector(mAllPropertyFeatures.row(i));
    ptr->SetVisilibities(this->mVisibilities, this->mVisibleIndex);
    // std::cout << "visible feature = " <<
    // ptr->BuildVisibleFeatureVector().transpose() << std::endl; std::cout <<
    // "full feature = " << ptr->BuildFullFeatureVector().transpose() <<
    // std::endl; std::cout
    //     << "hans't been finished yet\n";
    // exit(0);
    // ptr->ReadFeatureVector(CalcPropertyFromIndices(mNextSampleIndices));
    // AddIndices();
    return ptr;
}

// bool tPhyPropertyManager::IsEnd() const
// {
//     SIM_ASSERT(false && "hasn't been finished yet");
//     return true;
//     // if (mFirstSample)
//     // {
//     //     return false;
//     // }
//     // else
//     // {
//     //     bool is_end = true;
//     //     for (int i = 0; i < tPhyProperty::mNumOfProperties; i++)
//     //     {
//     //         SIM_ASSERT((mNextSampleIndices[i] >= 0) &&
//     //                    (mNextSampleIndices[i] < mSamplePerProperty));
//     //         if (mNextSampleIndices[i] != 0)
//     //         {
//     //             is_end = false;
//     //             break;
//     //         }
//     //     }
//     //     return is_end;
//     // }
// }

// void tPhyPropertyManager::InitPropRange(const Json::Value &prop_range_json)
// {
//     mPropMin = tVectorXd::Zero(tPhyProperty::mNumOfProperties);
//     mPropMax = tVectorXd::Zero(tPhyProperty::mNumOfProperties);
//     mPropDefault = tVectorXd::Zero(tPhyProperty::mNumOfProperties);
//     for (int i = 0; i < tPhyProperty::mNumOfProperties; i++)
//     {
//         tVector3d feature_range = cJsonUtil::ReadVectorJson(
//                                       cJsonUtil::ParseAsValue(tPhyProperty::mPropertiesName[i],
//                                       prop_range_json)) .segment(0, 3);
//         mPropMin[i] = feature_range[0];
//         mPropDefault[i] = feature_range[1];
//         mPropMax[i] = feature_range[2];
//     }
//     SIM_ASSERT(prop_range_json.size() == tPhyProperty::mNumOfProperties);
//     std::cout << "[debug] raw prop min = " << mPropMin.transpose() <<
//     std::endl; std::cout << "[debug] raw prop max = " << mPropMax.transpose()
//     << std::endl; std::cout << "[debug] raw prop default = " <<
//     mPropDefault.transpose() << std::endl;
//     // exit(0);
// }

// void tPhyPropertyManager::AddIndices()
// {
//     mNextSampleIndices[tPhyProperty::mNumOfProperties - 1]++;
//     for (int idx = tPhyProperty::mNumOfProperties - 1; idx >= 0; idx--)
//     {
//         // division
//         if (idx >= 1)
//         {
//             mNextSampleIndices[idx - 1] += std::floor(mNextSampleIndices[idx]
//             / mSamplePerProperty);
//         }
//         mNextSampleIndices[idx] %= mSamplePerProperty;
//     }
// }

/**
 * \brief
 */
#include <cmath>
std::vector<double>
tPhyPropertyManager::CalcPropertyDiscreteRange(int idx) const
{
    std::vector<double> values(0);
    int gap = mSamples[idx] - 1;
    SIM_ASSERT(gap >= 0);
    double st_value = mPropMin[idx];
    for (int i = 0; i < mSamples[idx]; i++)
    {
        switch (this->mSampleMode)
        {
        case eSampleMode::LOG:
        {
            double min_log_value = std::log(mPropMin[idx]);
            double incremental =
                (gap > 0 ? (std::log(mPropMax[idx]) - std::log(mPropMin[idx])) /
                               (gap)
                         : 0) *
                i;
            values.push_back(std::exp(min_log_value + incremental));
            break;
        }
        case eSampleMode::LINEAR:
        {
            double min_value = mPropMin[idx];
            double incremental =
                (gap > 0 ? (mPropMax[idx] - mPropMin[idx]) / (gap) : 0) * i;
            values.push_back(min_value + incremental);
            break;
        }
        default:
            SIM_ERROR("unsupported prop {}", mSampleMode);
            break;
        }
    }

    return values;
}

// void tPhyPropertyManager::PrintIndices()
// {
//     // std::cout << "[debug] cur indices = ";
//     // for (auto &x : mNextSampleIndices)
//     //     std::cout << x;
//     // std::cout << std::endl;
// }

tVectorXd vec2eigen(std::vector<double> res)
{
    tVectorXd vec = tVectorXd::Zero(res.size());
    for (int i = 0; i < res.size(); i++)
    {
        vec[i] = res[i];
    }
    return vec;
}

typedef std::vector<std::vector<double>> double_vv;
#include <algorithm>
void tPhyPropertyManager::InitFeatures()
{
    // 1. construct all values' sets
    std::vector<std::vector<double>> property_values(0);
    for (int i = 0; i < tPhyProperty::mNumOfProperties; i++)
    {
        property_values.push_back(CalcPropertyDiscreteRange(i));
        // std::cout << "vec = " <<
        // vec2eigen(property_values[property_values.size() - 1]).transpose() <<
        // std::endl;
    }

    // 2. cartesian prod
    double_vv res = cMathUtil::CartesianProductVec(property_values);

    // 3. begin to remove duplicate configuration by exchangeable pairs

    for (auto pair : mExchangeablePairs)
    {
        int prop_id0 = pair.first, prop_id1 = pair.second;
        double_vv::iterator it = res.begin();
        while (it != res.end())
        {
            // 1. find whether this
            std::vector<double> new_item = *it;
            std::swap(new_item[prop_id1], new_item[prop_id0]);
            auto new_item_it = std::find(res.begin(), res.end(), new_item);
            if (new_item_it != res.end() && new_item_it != it)
            {
                // std::cout << "from " << vec2eigen(*it).transpose() << "
                // delete " << vec2eigen(new_item).transpose() << std::endl;
                res.erase(new_item_it);
            }
            it++;
        }
    }

    mAllPropertyFeatures.noalias() =
        tMatrixXd::Zero(res.size(), tPhyProperty::mNumOfProperties);

    for (int i = 0; i < res.size(); i++)
    {
        for (int j = 0; j < res[i].size(); j++)
        {
            mAllPropertyFeatures(i, j) = res[i][j];
        }
    }

    // std::cout << "features = \n"
    //           << mAllPropertyFeatures << std::endl;
    // exit(0);
    // std::cout << "num of features = " << GetNumOfProperties() << std::endl;
    // exit(0);
}
int tPhyPropertyManager::GetNumOfProperties() const
{
    return mAllPropertyFeatures.rows();
}
#endif