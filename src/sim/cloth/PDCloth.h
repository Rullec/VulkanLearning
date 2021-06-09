#pragma once
#include "sim/cloth/BaseCloth.h"
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCholesky>

class cPDCloth : public cBaseCloth
{
public:
    cPDCloth();
    virtual ~cPDCloth();
    virtual void Init(const Json::Value &conf);
    virtual void UpdatePos(double dt) override;

protected:
    int mMaxSteps_Opt; // iterations used in fast simulation
    bool mEnableBending;
    double mBendingStiffness; // bending stiffness

    tVectorXd CalcNextPosition() const;
    // void InitVarsOptImplicit();
    void InitVarsOptImplicitSparse();
    void InitVarsOptImplicitSparseFast();
    virtual void InitGeometry(const Json::Value &conf) override final;
    const tEigenArr<tTriplet> &GetStretchTriplet() const;
    void AddBendTriplet(tEigenArr<tTriplet> &) const;
    // tMatrixXd J, I_plus_dt2_Minv_L_inv; // vars used in fast simulation
    tSparseMat J_sparse,
        I_plus_dt2_Minv_L_sparse; // vars used in fast simulation
    Eigen::SparseLU<tSparseMat> I_plus_dt2_Minv_L_sparse_solver;
    Eigen::SparseLU<tSparseMat> I_plus_dt2_Minv_L_sparse_fast_solver;
    tSparseMat I_plus_dt2_Minv_L_sparse_fast; // vars used in fast
};