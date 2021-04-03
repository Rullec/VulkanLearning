#pragma once
#include "Scene.h"
#include "utils/MathUtil.h"
struct tVertex
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    tVertex();
    double mMass;
    tVector mPos;
    tVector2f
        muv; // "texture" coordinate 2d, it means the plane coordinate for a vertex over a cloth, but now the texture in rendering
    tVector mColor;
};
struct tSpring
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    tSpring();
    double mRawLength;
    double mK;
    int mId0, mId1;
};

namespace Json
{
class Value;
};

class cSimScene : public cScene
{
public:
    enum eIntegrationScheme
    {
        SEMI_IMPLICIT = 0,
        IMPLICIT,
        NUM_OF_INTEGRATION_SCHEMES
    };
    explicit cSimScene();
    ~cSimScene();
    virtual void Init(const std::string &conf_path) override final;
    virtual void Update(double dt) override final;
    virtual void Reset() override final;
    const tVectorXf &GetTriangleDrawBuffer();
    const tVectorXf &GetEdgesDrawBuffer();

protected:
    eIntegrationScheme mScheme;
    double mClothWidth; // a square cloth
    double mClothMass;  // cloth mass
    int mSubdivision;   // division number along with the line
    double mStiffness;  // K
    tVectorXf mTriangleDrawBuffer,
        mEdgesDrawBuffer; // buffer to triangle buffer drawing (should use index buffer to improve the velocity)
    tEigenArr<tVertex *> mVertexArray; // vertices info
    tEigenArr<tSpring *> mSpringArray; // springs info
    tVectorXd mIntForce;               // internal force
    tVectorXd mExtForce;               // external force
    tVectorXd mInvMassMatrixDiag;      // diag inv mass matrix
    tVectorXd mXpre, mXcur; // previous node position & current node position
    std::vector<int> mFixedPointIds; // fixed constraint point
    void InitGeometry();             // discretazation from square cloth to
    void ClearForce();               // clear all forces
    void CalcInvMassMatrix() const;  // inv mass mat
    void CalcExtForce(tVectorXd &ext_force) const;
    void CalcIntForce(const tVectorXd &xcur, tVectorXd &int_force) const;
    tVectorXd
    CalcNextPositionSemiImplicit() const; // calculate xnext by semi implicit
    void UpdatePreNodalPosition(const tVectorXd &xpre);
    void UpdateCurNodalPosition(const tVectorXd &xcur);
    void CalcTriangleDrawBuffer(); //
    void CalcEdgesDrawBuffer();    //
    void GetVertexRenderingData();
    int GetNumOfVertices() const;
    int GetNumOfFreedom() const;
    void CalcNodePositionVector(tVectorXd &pos) const;
    void InitConstraint(const Json::Value &root);

    // implicit methods
    tVectorXd CalcNextPositionImplicit();
    void CalcGxImplicit(const tVectorXd &xcur, tVectorXd &Gx,
                        tVectorXd &fint_buf, tVectorXd &fext_buf) const;
    void CalcdGxdxImplicit(const tVectorXd &xcur, tMatrixXd &Gx) const;
    void TestdGxdxImplicit(const tVectorXd &x0, const tMatrixXd &Gx_ana);

    void PushState(const std::string &name) const;
    void PopState(const std::string &name);
};