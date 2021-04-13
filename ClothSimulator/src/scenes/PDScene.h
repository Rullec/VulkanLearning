#pragma once
#include "SimScene.h"

/**
 * \brief           projective dynamic simulation (stretch + bending)
*/
class cPDScene : public cSimScene
{
public:
    explicit cPDScene();
    virtual void Init(const std::string &conf_path) override;
    virtual void Update(double dt) override;
    virtual void Reset() override;

protected:
    int mMaxSteps_Opt; // iterations used in fast simulation

    tVectorXd CalcNextPositionOptImplicit() const;
    // void InitVarsOptImplicit();
    void InitVarsOptImplicitSparse();
    // tMatrixXd J, I_plus_dt2_Minv_L_inv; // vars used in fast simulation
    tSparseMat J_sparse,
        I_plus_dt2_Minv_L_sparse; // vars used in fast simulation
    Eigen::SparseLU<tSparseMat> I_plus_dt2_Minv_L_sparse_solver;
    virtual void InitGeometry(const Json::Value &conf) override;
    virtual void UpdateSubstep() override final;
};