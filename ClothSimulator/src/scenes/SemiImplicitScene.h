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
    double mStiffness; // K

    virtual void InitGeometry(const Json::Value &conf) override final;
    virtual void InitConstraint(const Json::Value &root) override final;

    tVectorXd
    CalcNextPositionSemiImplicit() const; // calculate xnext by semi implicit

    // optimization implicit methods (fast simulation)
    void PushState(const std::string &name) const;
    void PopState(const std::string &name);
    virtual void UpdateSubstep() override final;
};