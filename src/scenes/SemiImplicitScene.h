#pragma once
#include "SimScene.h"

/**
 * \brief           simulation scene for mass-spring system
*/
class cSemiImplicitScene : public cSimScene
{
public:
    explicit cSemiImplicitScene();
    ~cSemiImplicitScene();
    virtual void Init(const std::string &conf_path) override;
    virtual void Update(double dt) override;
    virtual void Reset() override;

protected:
    double mStiffness;        // K
    double mBendingStiffness; // bending stiffness
    bool mEnableQBending;     // enable Q bending
    tSparseMat mBendingHessianQ;
    virtual void InitGeometry(const Json::Value &conf) override final;
    virtual void InitConstraint(const Json::Value &root) override final;
    void InitBendingHessian();

    tVectorXd
    CalcNextPositionSemiImplicit() const; // calculate xnext by semi implicit
    virtual void CalcIntForce(const tVectorXd &xcur, tVectorXd &int_force) const override final;
    virtual void UpdateSubstep() override final;
};