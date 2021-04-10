#pragma
#include "SimScene.h"
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>

/**
 * \brief           simulation scene for mass-spring system
*/
struct tEdge;
class cMSScene : public cSimScene
{
public:
    explicit cMSScene();
    ~cMSScene();
    virtual void Init(const std::string &conf_path) override;
    virtual void Update(double dt) override;
    virtual void Reset() override;
    virtual void RayCast(tRay *ray) override final;

protected:
    int mMaxNewtonIters; // max newton iterations in implicit integration
    double mStiffness;   // K
    tVectorXd mInvMassMatrixDiag; // diag inv mass matrix

    virtual void InitGeometry(const Json::Value &conf) override final;
    virtual void InitConstraint(const Json::Value &root) override final;
    // derived methods
    void CalcDampingForce(const tVectorXd &vel, tVectorXd &damping) const;
    void CalcIntForce(const tVectorXd &xcur, tVectorXd &int_force) const;

    tVectorXd
    CalcNextPositionSemiImplicit() const; // calculate xnext by semi implicit

    // implicit methods
    tVectorXd CalcNextPositionImplicit();
    void CalcGxImplicit(const tVectorXd &xcur, tVectorXd &Gx,
                        tVectorXd &fint_buf, tVectorXd &fext_buf,
                        tVectorXd &fdamp_buffer) const;

    void CalcdGxdxImplicit(const tVectorXd &xcur, tMatrixXd &Gx) const;
    void CalcdGxdxImplicitSparse(const tVectorXd &xcur, tSparseMat &Gx) const;
    void TestdGxdxImplicit(const tVectorXd &x0, const tMatrixXd &Gx_ana);

    // optimization implicit methods (fast simulation)
    int mMaxSteps_Opt; // iterations used in fast simulation
    tVectorXd CalcNextPositionOptImplicit() const;
    tVectorXd CalcNextPositionOptImplicitSparse() const;
    // void InitVarsOptImplicit();
    void InitVarsOptImplicitSparse();
    // tMatrixXd J, I_plus_dt2_Minv_L_inv; // vars used in fast simulation
    tSparseMat J_sparse,
        I_plus_dt2_Minv_L_sparse; // vars used in fast simulation
    // Eigen::SimplicialLDLT<tSparseMat> I_plus_dt2_Minv_L_sparse_solver;
    Eigen::SparseLU<tSparseMat> I_plus_dt2_Minv_L_sparse_solver;
    // Eigen::SparseQR<tSparseMat> I_plus_dt2_Minv_L_sparse_solver;
    // Eigen::SimplicialLLT<tSparseMat> I_plus_dt2_Minv_L_sparse_solver;
    // Eigen::ConjugateGradient<tSparseMat> I_plus_dt2_Minv_L_sparse_solver;
    void PushState(const std::string &name) const;
    void PopState(const std::string &name);
    virtual void UpdateSubstep() override final;
    int GetNumOfSprings() const;
};