#pragma once
#include "Scene.h"
#include "utils/DefUtil.h"
#include "utils/MathUtil.h"

namespace Json
{
class Value;
};

enum eSceneType
{
    // MS means mass-spring system
    SCENE_SEMI_IMPLICIT = 0,
    SCENE_IMPLICIT,
    SCENE_PROJECTIVE_DYNAMIC, // see Liu Et al, "Fast simulation of mass spring
                              // system", equivalent to "optimization implicit
                              // euler"
    SCENE_POSITION_BASED_DYNAMIC, // trimesh modeling, position based dynamics
    SCENE_BARAFF, // trimesh modeling, baraff 98 siggraph "large step for cloth
                  // simulation"
    SCENE_SE,     // style 3d engine
    SCENE_SYN_DATA,     // synthetic train scene
    SCENE_PROCESS_DATA, // process train scene
    SCENE_MESH_VIS,     // mesh visualization scene
    NUM_OF_SCENE_TYPES
};

struct tVertex;
struct tEdge;
struct tTriangle;
struct tRay;
struct tPerturb;
// class cDrawScene;
SIM_DECLARE_CLASS_AND_PTR(cKinematicBody)
SIM_DECLARE_CLASS_AND_PTR(cRaycaster)
SIM_DECLARE_CLASS_AND_PTR(cCollisionDetecter)
SIM_DECLARE_CLASS_AND_PTR(cBaseCloth)
class cSimScene : public cScene
{
public:
    inline static const std::string ENABLE_PROFLINE_KEY = "enable_profiling",
                                    ENABLE_OBSTACLE_KEY = "enable_obstacle",
                                    OBSTACLE_CONF_KEY = "obstacle_conf",
                                    ENABLE_COLLISION_DETECTION_KEY =
                                        "enable_collision_detection";

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
    virtual void UpdatePerturb(const tVector &camera_pos, const tVector &dir);
    virtual void CursorMove(int xpos, int ypos);
    virtual void MouseButton(int button, int action, int mods);
    virtual void Key(int key, int scancode, int action, int mods);
    void RayCastScene(const tRay *ray, tTriangle **selected_triangle,
                      int &selected_triangle_id,
                      tVector &ray_cast_position) const;

protected:
    tPerturb *mPerturb;

    // eSceneType mSceneType;
    bool mEnableProfiling;
    bool mEnableObstacle; // using obstacle?
    bool mEnableCollisionDetection;
    std::vector<cKinematicBodyPtr>
        mObstacleList;        // obstacle for cloth simulation
    cRaycasterPtr mRaycaster; // raycaster
    // double mClothWidth;           // a square cloth
    // double mClothMass;            // cloth mass
    // tVector mClothInitPos;        //
    // int mSubdivision;             // division number along with the line

    tVectorXf mTriangleDrawBuffer,
        mEdgesDrawBuffer; // buffer to triangle buffer drawing (should use index
                          // buffer to improve the velocity)
    std::vector<tRay *> mRayArray;
    // std::vector<tVertex *> mVertexArray;     // total vertices
    // std::vector<tEdge *> mEdgeArray;         // total edges
    // std::vector<tTriangle *> mTriangleArray; // total triangles

    // cCollisionDetecterPtr mColDetecter; // collision detecter
    cBaseClothPtr mCloth;

    // base methods
    void CalcDampingForce(const tVectorXd &vel, tVectorXd &damping) const;
    virtual void InitDrawBuffer();
    virtual void InitRaycaster();

    void ClearForce(); // clear all forces

    virtual void CalcTriangleDrawBuffer(); //
    virtual void CalcEdgesDrawBuffer();    //
    void GetVertexRenderingData();
    int GetNumOfVertices() const;
    int GetNumOfFreedom() const;
    int GetNumOfEdges() const;
    int GetNumOfTriangles() const;
    void CalcNodePositionVector(tVectorXd &pos) const;
    // virtual void InitConstraint(const Json::Value &root);
    // virtual void UpdateCurNodalPosition(const tVectorXd &xcur);
    // virtual void UpdateSubstep() = 0;

    // virtual void CreatePerturb(tRay *ray);

    virtual void CreateObstacle(const Json::Value &conf);
    virtual void CreateCollisionDetecter();
    bool mPauseSim;
    virtual void PauseSim();
    void CreateCloth(const Json::Value &conf);
    // virtual int GetNumOfTriangles() const;
};