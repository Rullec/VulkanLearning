#pragma once
#include "SimScene.h"
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>
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
    bool mEnableBending;
    double mBendingStiffness; //bending stiffness
    tVectorXd CalcNextPosition() const;
    // void InitVarsOptImplicit();
    void InitVarsOptImplicitSparse();
    void InitVarsOptImplicitSparseFast();
    const tEigenArr<tTriplet> &GetStretchTriplet() const;
    void AddBendTriplet(tEigenArr<tTriplet> &) const;
    // tMatrixXd J, I_plus_dt2_Minv_L_inv; // vars used in fast simulation
    tSparseMat J_sparse,
        I_plus_dt2_Minv_L_sparse; // vars used in fast simulation
    Eigen::SparseLU<tSparseMat> I_plus_dt2_Minv_L_sparse_solver;
    Eigen::SparseLU<tSparseMat> I_plus_dt2_Minv_L_sparse_fast_solver;

    // wrong result
    // Eigen::SimplicialLLT<tSparseMat> I_plus_dt2_Minv_L_sparse_solver;
    // Eigen::SimplicialLLT<tSparseMat> I_plus_dt2_Minv_L_sparse_fast_solver;
    // Eigen::SimplicialLDLT<tSparseMat> I_plus_dt2_Minv_L_sparse_solver;
    // Eigen::SimplicialLDLT<tSparseMat> I_plus_dt2_Minv_L_sparse_fast_solver;
    // Eigen::ConjugateGradient<tSparseMat> I_plus_dt2_Minv_L_sparse_solver;
    // Eigen::ConjugateGradient<tSparseMat> I_plus_dt2_Minv_L_sparse_fast_solver;
    // Eigen::BiCGSTAB<tSparseMat> I_plus_dt2_Minv_L_sparse_solver;
    // Eigen::BiCGSTAB<tSparseMat> I_plus_dt2_Minv_L_sparse_fast_solver;

    // fast impl
    tSparseMat I_plus_dt2_Minv_L_sparse_fast; // vars used in fast simulation
    virtual void InitGeometry(const Json::Value &conf) override;
    virtual void UpdateSubstep() override final;
};