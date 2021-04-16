#ifdef _WIN32
#pragma once
#include "SimScene.h"

/**
 * \brief              support style3d engine
*/
namespace StyleEngine
{
    class SePiece;
    class SeScene;
    class SeDraggedPoints;
};
class cLinctexScene : public cSimScene
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    explicit cLinctexScene();
    virtual void Init(const std::string &path) override;
    virtual void Update(double dt) override final;
    virtual ~cLinctexScene();

protected:
    virtual void UpdateSubstep() override final;
    virtual void InitConstraint(const Json::Value &root) override final;
    void AddPiece();
    void ReadVertexPosFromEngine();
    virtual bool CreatePerturb(tRay *ray) override final;
    virtual void ReleasePerturb() override final;
    virtual void UpdatePerturb();
    std::shared_ptr<StyleEngine::SePiece> mCloth;
    std::shared_ptr<StyleEngine::SeDraggedPoints> mDragPt;
    std::shared_ptr<StyleEngine::SeScene> mSeScene;
    struct tPhyProperty
    {
        double mStretchWarp, mStretchWeft;
        double mBendingWarp, mBendingWeft;
        void Init(const Json::Value &conf);
    } mClothProp;
};
#endif