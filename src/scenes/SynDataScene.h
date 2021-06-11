#ifdef _WIN32
#pragma once
#include "LinctexScene.h"
#include "SimScene.h"
#include "utils/DefUtil.h"
#include <string>
/**
 * \brief           scene for synthetic data
 */
SIM_DECLARE_CLASS_AND_PTR(cLinctexScene);
SIM_DECLARE_CLASS_AND_PTR(cLinctexCloth);
SIM_DECLARE_CLASS_AND_PTR(tPhyProperty);
SIM_DECLARE_CLASS_AND_PTR(tPhyPropertyManager);
class cSynDataScene : public cSimScene
{
public:
    inline static const std::string ENABLE_DRAW_KEY = "enable_draw";
    explicit cSynDataScene();
    virtual void Init(const std::string &conf_path) override;
    virtual void Update(double dt) override;
    virtual void Reset() override;
    virtual const tVectorXf &GetTriangleDrawBuffer() override;
    virtual const tVectorXf &GetEdgesDrawBuffer() override;
    static eSceneType BuildSceneType(const std::string &str);
    virtual bool CreatePerturb(tRay *ray) override;
    virtual void CursorMove(int xpos, int ypos) override;
    virtual void MouseButton(int button, int action, int mods) override;
    virtual void Key(int key, int scancode, int action, int mods) override;

protected:
    struct tSyncDataNoise
    {
        tSyncDataNoise(const Json::Value &value);
        bool mEnableNoise;
        int mNumOfSamplesPerProp;
        // bool mEnableInitYRotation;
        // bool mEnableFoldNoise;
        // bool mEnableInitYPosNoise;
        // double mInitYPosNoiseStd;
        // double mFoldCoef;
        bool mEnableLowFreqNoise;
        double mMaxFoldAmp;
        int mMinFoldNum, mMaxFoldNum;
        // void ApplyNoise(std::vector<tVertex *> &vertex_array);
        // protected:
        // tEigenArr<tMatrix> GenerateAugmentTransform() const;
        // tEigenArr<tMatrix> mTrans;
    };
    bool mEnableDraw; // enable drawing when sampling (for debug purpose)
    std::shared_ptr<tSyncDataNoise> mSynDataNoise;
    cLinctexScenePtr mLinScene;
    cLinctexClothPtr mLinCloth;
    std::string mDefaultConfigPath;      // config used to build simulation
    tPhyPropertyManagerPtr mPropManager; // physical property manager
    double mConvergenceThreshold;
    std::string mExportDataDir;

    bool mEnableDataCleaner;
    double mDataCleanerThreshold;
    void RunSimulation(tPhyPropertyPtr props);
    void ApplyNoiseIfPossible();
    void InitExportDataDir();
    bool CheckDuplicateWithDataSet() const;
    void OfflineSampling();
    tVectorXd buffer0, buffer1;
};
#endif