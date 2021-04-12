#pragma once
#include "Scene.h"
#include "utils/MathUtil.h"

namespace Json
{
class Value;
};

enum eIntegrationScheme
{
    // MS means mass-spring system
    MS_SEMI_IMPLICIT = 0,
    MS_IMPLICIT,
    MS_OPT_IMPLICIT, // see Liu Et al, "Fast simulation of mass spring system", equivalent to "optimization implicit euler"
    TRI_POSITION_BASED_DYNAMIC, // trimesh modeling, position based dynamics
    TRI_PROJECTIVE_DYNAMIC,     // trimesh
    TRI_BARAFF, // trimesh modeling, baraff 98 siggraph "large step for cloth simulation"
    NUM_OF_INTEGRATION_SCHEMES
};

struct tVertex;
struct tEdge;
struct tTriangle;
struct tRay;
struct tPerturb;
class cDrawScene;
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
    virtual void RayCast(tRay *ray);
    virtual void CursorMove(cDrawScene *draw_scene, int xpos, int ypos);
    virtual void MouseButton(cDrawScene *draw_scene, int button, int action,
                             int mods);

protected:
    tPerturb *mPerturb;
    tVectorXd mInvMassMatrixDiag; // diag inv mass matrix
    std::string mGeometryType;
    eIntegrationScheme mScheme;
    bool mEnableProfiling;
    // double mClothWidth;           // a square cloth
    // double mClothMass;            // cloth mass
    // tVector mClothInitPos;        //
    // int mSubdivision;             // division number along with the line
    double mDamping;              // damping coeff
    double mIdealDefaultTimestep; // default substep dt
    tVectorXf mTriangleDrawBuffer,
        mEdgesDrawBuffer; // buffer to triangle buffer drawing (should use index buffer to improve the velocity)
    std::vector<tRay *> mRayArray;
    std::vector<tVertex *> mVertexArray;     // vertices info
    std::vector<tEdge *> mEdgeArray;         // springs info
    std::vector<tTriangle *> mTriangleArray; // triangles info
    tVectorXd mIntForce;                     // internal force
    tVectorXd mExtForce;                     // external force
    tVectorXd mDampingForce;                 // external force

    tVectorXd mXpre, mXcur; // previous node position & current node position
    std::vector<int> mFixedPointIds; // fixed constraint point

    // base methods
    void CalcDampingForce(const tVectorXd &vel, tVectorXd &damping) const;
    virtual void InitGeometry(
        const Json::Value &conf); // discretazation from square cloth to
    void ClearForce();            // clear all forces
    virtual void CalcIntForce(const tVectorXd &xcur, tVectorXd &int_force) const;
    virtual void CalcExtForce(tVectorXd &ext_force) const;
    virtual void CalcTriangleDrawBuffer(); //
    virtual void CalcEdgesDrawBuffer();    //
    void GetVertexRenderingData();
    int GetNumOfVertices() const;
    int GetNumOfFreedom() const;
    int GetNumOfEdges() const;
    void CalcNodePositionVector(tVectorXd &pos) const;
    virtual void InitConstraint(const Json::Value &root);
    void UpdateCurNodalPosition(const tVectorXd &xcur);
    virtual void UpdateSubstep() = 0;
};