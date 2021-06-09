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
    std::vector<std::pair<int, int>> mExchangeablePairs;
    void InitExchangeablePairs(const Json::Value &conf);
    void InitFeatures();
    std::vector<double> CalcPropertyDiscreteRange(int idx) const;
};
#endif