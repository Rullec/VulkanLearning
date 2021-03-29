#pragma once
#include "Scene.h"
#include "utils/MathUtil.h"
struct tVertex
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    tVertex();
    double mMass;
    tVector mPos;
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

class cSimScene : public cScene
{
public:
    explicit cSimScene();
    ~cSimScene();
    virtual void Init(const std::string &conf_path) override final;
    virtual void Update(double dt) override final;
    virtual void Reset() override final;

protected:
    double mClothWidth;                // a square cloth
    int mSubdivision;                  // division number along with the line
    tEigenArr<tVertex *> mVertexArray; // vertices info
    tEigenArr<tSpring *> mSpringArray; // springs info
    tVectorXd mIntForce;               // internal force
    tVectorXd mExtForce;               // external force
    tVectorXd mInvMassMatrixDiag;      // diag inv mass matrix

    void InitGeometry();            // discretazation from square cloth to
    void ClearForce();              // clear all forces
    void CalcInvMassMatrix() const; // inv mass mat
    void CalcExtForce() const;
    void CalcIntForce() const;
    void CalcNextPosition();        // forward simulation
    void CalcVertexRenderingData(); //
    void GetVertexRenderingData();
    int GetNumOfVertices() const;
    int GetNumOfFreedom() const;
};