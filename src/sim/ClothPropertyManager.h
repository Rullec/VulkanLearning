#ifdef _WIN32
#include "utils/DefUtil.h"
#include "utils/MathUtil.h"
namespace Json
{
    class Value;
};
SIM_DECLARE_CLASS_AND_PTR(tPhyProperty);
class tPhyPropertyManager
{
public:
    explicit tPhyPropertyManager(const Json::Value &conf);
    tPhyPropertyPtr GetNextProperty();
    bool IsEnd() const;
    void PrintIndices();

protected:
    tVectorXd mPropMin, mPropMax, mPropDefault;
    int mSamples;
    std::vector<int> mNextSampleIndices; // the indices for next property
    enum eSampleMode
    {
        LINEAR = 0,
        LOG
    };
    bool mFirstSample;
    eSampleMode mSampleMode;
    void InitPropRange(const Json::Value &feature_range_json);
    void AddIndices();
    tVectorXd CalcPropertyFromIndices(const std::vector<int> &indices) const;
};
#endif