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
    virtual void Update(double dt) override;
    virtual void Reset() override;

protected:
    int mMaxNewtonIters; // max newton iterations in implicit integration
    // implicit methods
    tVectorXd CalcNextPositionImplicit();
    void CalcGxImplicit(const tVectorXd &xcur, tVectorXd &Gx,
                        tVectorXd &fint_buf, tVectorXd &fext_buf,
                        tVectorXd &fdamp_buffer) const;

    void CalcdGxdxImplicit(const tVectorXd &xcur, tMatrixXd &Gx) const;
    void CalcdGxdxImplicitSparse(const tVectorXd &xcur, tSparseMat &Gx) const;
    void TestdGxdxImplicit(const tVectorXd &x0, const tMatrixXd &Gx_ana);
};