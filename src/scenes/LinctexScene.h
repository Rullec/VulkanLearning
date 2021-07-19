#ifdef _WIN32
#pragma once
#include "SimScene.h"
#include <string>

/**
 * \brief              support style3d engine
 */
namespace StyleEngine
{
class SePiece;
class SeScene;
class SeDraggedPoints;
}; // namespace StyleEngine
SIM_DECLARE_CLASS_AND_PTR(tPhyProperty);
SIM_DECLARE_CLASS_AND_PTR(cLinctexCloth);
SIM_DECLARE_CLASS_AND_PTR(cMessageCallback);

class cLinctexScene : public cSimScene
{
public:
    inline static const std::string SE_SIM_PLATFORM_KEY = "se_sim_platform",
                                    SE_ENABLE_COLLISION_KEY =
                                        "se_enable_collision";
    explicit cLinctexScene();
    virtual void Init(const std::string &path) override;
    virtual void Update(double dt) override final;
    virtual ~cLinctexScene();
    virtual bool CreatePerturb(tRay *ray) override final;
    virtual void ReleasePerturb() override final;
    virtual void Reset() override final;
    cLinctexClothPtr GetLinctexCloth() const;
    int GetCurrentFrame() const;
    void Start();
    static double CalcSimDiff(const tVectorXd &v0, const tVectorXd &v1);
    // external cloth property settings
    // virtual void SetSimProperty(const tPhyPropertyPtr &prop);
    // virtual void ApplyTransform(const tMatrix &trans);
    // virtual void ApplyNoise(bool enable_y_random_rotation,
    //                         double &rotation_angle, bool enable_y_random_pos,
    //                         const double random_ypos_std);
    // virtual void ApplyFoldNoise(const tVector3d &principle_noise,
    //                             const double a);
    // virtual void ApplyMultiFoldsNoise(int num_of_folds, double max_amp);
    // virtual tPhyPropertyPtr GetSimProperty() const;
    // virtual const tVectorXd &GetClothFeatureVector() const;
    // virtual int GetClothFeatureSize() const;
    virtual void Key(int key, int scancode, int action, int mods);
    // virtual tVector CalcCOM() const;
    virtual void End();
    virtual void SetEnableCheckSimulationDiff(bool val);
    virtual void UpdateCurTimeRec(); // update current time record
    virtual void CheckOutputIfPossible(); // update current time record
protected:
    void AddPiece();
    virtual void UpdatePerturb();
    void NetworkInferenceFunction();

    virtual void PauseSim() override;
    std::shared_ptr<StyleEngine::SeDraggedPoints> mDragPt;
    std::shared_ptr<StyleEngine::SeScene> mSeScene;

    bool mEngineStart; // start the engine or not

    bool mEnableNetworkInferenceMode; // if it's true, the simulation is running
                                      // for the DNN 's inference proceduce
    double mNetworkInfer_ConvThreshold; // used in network inference mode, the
                                        // convergence threshold for diff norm
                                        // between nodal positions
    std::string
        mNetworkInfer_OutputPath; // used in network inference mode, output path
                                  // of simulation result (nodal positions)
    int mNetworkInfer_MinIter; // used in network inference mode, the minimium
                               // iteration times before convergence
    int mNetworkInfer_CurIter; // used in network inference mode, cur iterations
    tVectorXd mPreviosFeature; // used in network inference mode, previous nodal
    // position vector
    cMessageCallbackPtr mMstPtr;
    bool mEnableCheckSimulationDiff;          // enable check simulation diff
    double mCheckSimulationDiffElaspedSecond; // check
    virtual void CreateCloth(const Json::Value &conf) override final;

    virtual void CreateObstacle(const Json::Value &conf) override final;
};
#endif