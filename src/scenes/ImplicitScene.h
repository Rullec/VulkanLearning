#pragma once
#include "SimScene.h"
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>

class cImplicitScene : public cSimScene
{
public:
    explicit cImplicitScene();
    ~cImplicitScene();
    virtual void Init(const std::string &conf_path) override;
    virtual void Reset() override;

protected:
    int mMaxNewtonIters; // max newton iterations in implicit integration
    double mStiffness;   // spring stiffness
    // implicit methods
    tVectorXd CalcNextPositionImplicit();
    void CalcGxImplicit(const tVectorXd &xcur, tVectorXd &Gx,
                        tVectorXd &fint_buf, tVectorXd &fext_buf,
                        tVectorXd &fdamp_buffer) const;
    virtual void InitGeometry(const Json::Value &conf) override final;
    void CalcdGxdxImplicit(const tVectorXd &xcur, tMatrixXd &Gx) const;
    void CalcdGxdxImplicitSparse(const tVectorXd &xcur, tSparseMat &Gx) const;
    void TestdGxdxImplicit(const tVectorXd &x0, const tMatrixXd &Gx_ana);
    virtual void UpdateSubstep() override final;
};