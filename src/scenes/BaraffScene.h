#pragma once
#include "SimScene.h"

/**
 * \brief               Baraff 98 SIGGRAPH "large step of cloth simulation"
*/
class cBaraffScene : public cSimScene
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    explicit cBaraffScene();
    virtual void Init(const std::string &conf_path) override;
    virtual ~cBaraffScene();

protected:
    virtual void UpdateSubstep() override final;
};