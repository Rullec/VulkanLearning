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
};
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
    virtual tPhyPropertyPtr GetSimProperty() const;
    virtual const tVectorXd &GetClothFeatureVector() const;
    virtual int GetClothFeatureSize() const;

protected:
    virtual void UpdateSubstep() override final;
    virtual void InitConstraint(const Json::Value &root) override final;
    void AddPiece();
    void ReadVertexPosFromEngine();
    virtual void UpdateCurNodalPosition(const tVectorXd &xcur) override final;

    virtual void UpdatePerturb();
    virtual void CreateObstacle(const Json::Value &conf);
    tVectorXd mClothFeature;
    virtual void InitClothFeatureVector();
    virtual void UpdateClothFeatureVector();
    std::shared_ptr<StyleEngine::SePiece> mCloth;
    std::shared_ptr<StyleEngine::SeDraggedPoints> mDragPt;
    std::shared_ptr<StyleEngine::SeScene> mSeScene;
    tPhyPropertyPtr mClothProp; // cloth property
    bool mEngineStart;          // start the engine or not
};
#endif