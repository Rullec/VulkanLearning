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

namespace Json
{
    class Value;
};

enum eIntegrationScheme
{
    // MS means mass-spring system
    MS_SEMI_IMPLICIT = 0,
    MS_IMPLICIT,
    MS_OPT_IMPLICIT,            // see Liu Et al, "Fast simulation of mass spring system", equivalent to "optimization implicit euler"
    TRI_POSITION_BASED_DYNAMIC, // trimesh modeling, position based dynamics
    TRI_BARAFF,                 // trimesh modeling, baraff 98 siggraph "large step for cloth simulation"
    NUM_OF_INTEGRATION_SCHEMES
};

class cSimScene : public cScene
{
public:
    explicit cSimScene();
    ~cSimScene();
    virtual void Init(const std::string &conf_path) override;
    virtual void Update(double dt) override;
    virtual void Reset() override;
    const tVectorXf &GetTriangleDrawBuffer();
    const tVectorXf &GetEdgesDrawBuffer();
    static eIntegrationScheme BuildIntegrationScheme(const std::string &str);

protected:
    eIntegrationScheme mScheme;
    double mClothWidth;           // a square cloth
    double mClothMass;            // cloth mass
    tVector mClothInitPos;      //
    int mSubdivision;             // division number along with the line
    double mStiffness;            // K
    double mDamping;              // damping coeff
    double mIdealDefaultTimestep; // default substep dt
    tVectorXf mTriangleDrawBuffer,
        mEdgesDrawBuffer; // buffer to triangle buffer drawing (should use index buffer to improve the velocity)

    tEigenArr<tVertex *> mVertexArray; // vertices info
    tVectorXd mIntForce;               // internal force
    tVectorXd mExtForce;               // external force
    tVectorXd mDampingForce;           // external force

    tVectorXd mXpre, mXcur;          // previous node position & current node position
    std::vector<int> mFixedPointIds; // fixed constraint point

    // base methods
    virtual void InitGeometry() = 0; // discretazation from square cloth to
    void ClearForce();               // clear all forces
    void CalcExtForce(tVectorXd &ext_force) const;
    virtual void CalcTriangleDrawBuffer(); //
    virtual void CalcEdgesDrawBuffer();    //
    void GetVertexRenderingData();
    int GetNumOfVertices() const;
    int GetNumOfFreedom() const;
    void CalcNodePositionVector(tVectorXd &pos) const;
    virtual void InitConstraint(const Json::Value &root);
    void UpdateCurNodalPosition(const tVectorXd &xcur);
    virtual void UpdateSubstep() = 0;
};