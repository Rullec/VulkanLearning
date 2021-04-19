#ifdef _WIN32
#include "utils/MathUtil.h"
namespace Json
{
    class Value;
};
struct tPhyProperty
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    double mStretchWarp;
    double mStretchWeft;
    double mBendingWarp;
    double mBendingWeft;
    inline static const int mNumOfProperties = 4;
    inline static const std::string mPropertiesName[tPhyProperty::mNumOfProperties] = {
        "stretch_warp",
        "stretch_weft",
        "bending_warp",
        "bending_weft"};
    void Init(const Json::Value &conf);
    tVectorXd BuildFeatureVector() const;
    void ReadFeatureVector(const tVectorXd &vec);

};
#endif