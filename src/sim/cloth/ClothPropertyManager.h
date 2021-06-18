#ifdef _WIN32
#include "utils/DefUtil.h"
#include "utils/MathUtil.h"
#include <utility>
namespace Json
{
class Value;
};
SIM_DECLARE_CLASS_AND_PTR(tPhyProperty);
class tPhyPropertyManager
{
public:
    inline static const std::string ENABLE_EXTERNAL_PROPERTY_SAMPLES_KEY = "enable_external_property_samples",
    EXTERNAL_PROPERTY_SAMPLES_PATH_KEY = "external_property_samples_path";
    explicit tPhyPropertyManager(const Json::Value &conf);
    tPhyPropertyPtr GetProperty(int idx);
    int GetNumOfProperties() const;

protected:
    tVectorXd mPropMin, mPropMax;
    tVectorXi mSamples;
    std::vector<bool> mVisibilities;
    tVectorXi mVisibleIndex;

    enum eSampleMode
    {
        LINEAR = 0,
        LOG
    };
    eSampleMode mSampleMode;
    tMatrixXd mAllPropertyFeatures;
    bool mEnableExternalPropertySamples;
    std::string mExternalPropertySamplesPath;
    std::vector<std::pair<int, int>> mExchangeablePairs;
    void InitExchangeablePairs(const Json::Value &conf);
    void InitFeaturesFromSampling();
    void InitFeaturesFromGivenFile();
    std::vector<double> CalcPropertyDiscreteRange(int idx) const;
};
#endif