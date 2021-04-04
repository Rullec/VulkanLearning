#pragma
#include "SimScene.h"

struct tSpring
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    tSpring();
    double mRawLength;
    double mK;
    int mId0, mId1;
};

/**
 * \brief           simulation scene for mass-spring system
*/
class cMSScene : public cSimScene
{
public:
    explicit cMSScene();
    ~cMSScene();
    virtual void Init(const std::string &conf_path) override;
    virtual void Update(double dt) override;
    virtual void Reset() override;

protected:
    int mMaxNewtonIters; // max newton iterations in implicit integration

    tEigenArr<tSpring *> mSpringArray; // springs info
    tVectorXd mInvMassMatrixDiag;      // diag inv mass matrix

    virtual void InitGeometry() override final;
    virtual void InitConstraint(const Json::Value &root) override final;
    // derived methods
    void CalcExtForce(tVectorXd &ext_force) const;
    void CalcDampingForce(const tVectorXd &vel, tVectorXd &damping) const;
    void CalcIntForce(const tVectorXd &xcur, tVectorXd &int_force) const;

    void CalcInvMassMatrix() const; // inv mass mat

    tVectorXd
    CalcNextPositionSemiImplicit() const; // calculate xnext by semi implicit

    // implicit methods
    tVectorXd CalcNextPositionImplicit();
    void CalcGxImplicit(const tVectorXd &xcur, tVectorXd &Gx,
                        tVectorXd &fint_buf, tVectorXd &fext_buf, tVectorXd &fdamp_buffer) const;

    void CalcdGxdxImplicit(const tVectorXd &xcur, tMatrixXd &Gx) const;
    void CalcdGxdxImplicitSparse(const tVectorXd &xcur, tSparseMat &Gx) const;
    void TestdGxdxImplicit(const tVectorXd &x0, const tMatrixXd &Gx_ana);

    void PushState(const std::string &name) const;
    void PopState(const std::string &name);
    virtual void UpdateSubstep() override final;
};