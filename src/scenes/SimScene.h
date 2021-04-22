#pragma once
#include "Scene.h"
#include "utils/MathUtil.h"
#include "utils/DefUtil.h"

namespace Json
{
    class Value;
};

enum eSceneType
{
    // MS means mass-spring system
    SCENE_SEMI_IMPLICIT = 0,
    SCENE_IMPLICIT,
    SCENE_PROJECTIVE_DYNAMIC,     // see Liu Et al, "Fast simulation of mass spring system", equivalent to "optimization implicit euler"
    SCENE_POSITION_BASED_DYNAMIC, // trimesh modeling, position based dynamics
    SCENE_BARAFF,                 // trimesh modeling, baraff 98 siggraph "large step for cloth simulation"
    SCENE_SE,                     // style 3d engine
    SCENE_SYN_DATA,
    NUM_OF_SCENE_TYPES
};

struct tVertex;
struct tEdge;
struct tTriangle;
struct tRay;
struct tPerturb;
class cDrawScene;
SIM_DECLARE_CLASS_AND_PTR(cKinematicBody)
class cSimScene : public cScene
{
public:
    cSimScene();
    ~cSimScene();
    virtual void Init(const std::string &conf_path) override;
    virtual void Update(double dt) override;
    virtual void UpdateRenderingResource();
    virtual void Reset() override;
    virtual const tVectorXf &GetTriangleDrawBuffer();
    virtual const tVectorXf &GetEdgesDrawBuffer();
    static eSceneType BuildSceneType(const std::string &str);
    virtual bool CreatePerturb(tRay *ray);
    virtual void ReleasePerturb();
    virtual void CursorMove(cDrawScene *draw_scene, int xpos, int ypos);
    virtual void MouseButton(cDrawScene *draw_scene, int button, int action,
                             int mods);
    virtual void Key(int key, int scancode, int action, int mods);

protected:
    double mClothWidth;
    double mClothMass;
    tPerturb *mPerturb;
    tVectorXd mInvMassMatrixDiag; // diag inv mass matrix
    std::string mGeometryType;
    eSceneType mScheme;
    bool mEnableProfiling;
    bool mEnableObstacle;        // using obstacle?
    cKinematicBodyPtr mObstacle; // obstacle for cloth simulation
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

    tVectorXd mXpre, mXcur;          // previous node position & current node position
    std::vector<int> mFixedPointIds; // fixed constraint point
    tVectorXd mClothInitPos;         // init position of the cloth
    // base methods
    void CalcDampingForce(const tVectorXd &vel, tVectorXd &damping) const;
    virtual void InitDrawBuffer();
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
    virtual void UpdateCurNodalPosition(const tVectorXd &xcur);
    virtual void UpdateSubstep() = 0;

    // virtual void CreatePerturb(tRay *ray);

    virtual void CreateObstacle(const Json::Value &conf);
};