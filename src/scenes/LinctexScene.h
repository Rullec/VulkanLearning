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

class cLinctexScene : public cSimScene
{
public:
    explicit cLinctexScene();
    virtual void Init(const std::string &path) override;
    virtual void Update(double dt) override final;
    virtual ~cLinctexScene();
    virtual bool CreatePerturb(tRay *ray) override final;
    virtual void ReleasePerturb() override final;
    virtual void Reset() override final;
    // external cloth property settings
    virtual void SetSimProperty(const tPhyPropertyPtr &prop);
    // virtual void ApplyTransform(const tMatrix &trans);
    virtual void ApplyNoise(bool enable_y_random_rotation,
                            double &rotation_angle, bool enable_y_random_pos,
                            const double random_ypos_std);
    virtual void ApplyFoldNoise(const tVector3d &principle_noise,
                                const double a);
    virtual void ApplyMultiFoldsNoise(int num_of_folds, double max_amp);
    virtual tPhyPropertyPtr GetSimProperty() const;
    virtual const tVectorXd &GetClothFeatureVector() const;
    virtual int GetClothFeatureSize() const;
    virtual void Key(int key, int scancode, int action, int mods);
    virtual tVector CalcCOM() const;
    virtual void End();

protected:
    virtual void UpdateSubstep() override final;
    virtual void InitConstraint(const Json::Value &root) override final;
    void AddPiece();
    void ReadVertexPosFromEngine();
    virtual void UpdateCurNodalPosition(const tVectorXd &xcur) override final;
    virtual void InitGeometry(const Json::Value &conf);
    virtual void UpdatePerturb();
    virtual void CreateObstacle(const Json::Value &conf);
    tVectorXd mClothFeature;
    virtual void InitClothFeatureVector();
    virtual void UpdateClothFeatureVector();
    void NetworkInferenceFunction();

    virtual void PauseSim() override;
    std::shared_ptr<StyleEngine::SePiece> mCloth;
    std::shared_ptr<StyleEngine::SeDraggedPoints> mDragPt;
    std::shared_ptr<StyleEngine::SeScene> mSeScene;
    tPhyPropertyPtr mClothProp; // cloth property
    bool mEngineStart;          // start the engine or not

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

    bool mEnableDumpGeometryInfo; // if true, we save the geometry information
                                  // after the initialization
    std::string mDumpGeometryInfoPath; // save path for initial geometry
};
#endif