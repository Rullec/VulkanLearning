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
    double mBendingBias;

    inline static const int mNumOfProperties = 5;
    inline static const std::string mPropertiesName[tPhyProperty::mNumOfProperties] = {
        "stretch_warp",
        "stretch_weft",
        "bending_warp",
        "bending_weft",
        "bending_bias"};

    virtual void Init(const Json::Value &conf); // normal init
    virtual tVectorXd BuildFullFeatureVector() const;
    virtual tVectorXd BuildVisibleFeatureVector() const;
    virtual void ReadFeatureVector(const tVectorXd &vec);
    static int GetFeatureIdx(std::string name);
    double GetFeature(std::string name) const;
};

struct tBatchProperty : public tPhyProperty
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    bool mVisibility[mNumOfProperties] = {};
    tVectorXi mVisibleFeatureIndex;
    int mNumOfVisibleFeature;
    void SetVisilibities(std::vector<bool> visibilities, const tVectorXi &visible_faeture_index);
    virtual tVectorXd BuildVisibleFeatureVector() const override;
};
#endif