#include "SimScene.h"
#include "Perturb.h"
#include "geometries/Primitives.h"
#include "geometries/Triangulator.h"
#include "sim/CollisionDetecter.h"
#include "sim/KinematicBody.h"
#include "sim/cloth/BaseCloth.h"
#include "utils/JsonUtil.h"
#include <iostream>

std::string gSceneTypeStr[eSceneType::NUM_OF_SCENE_TYPES] = {
    "sim", "se", "data_synthesis", "data_process", "mesh_vis"};

eSceneType cSimScene::BuildSceneType(const std::string &str)
{
    int i = 0;
    for (i = 0; i < eSceneType::NUM_OF_SCENE_TYPES; i++)
    {
        // std::cout << gSceneTypeStr[i] << std::endl;
        if (str == gSceneTypeStr[i])
        {
            break;
        }
    }

    SIM_ASSERT(i != eSceneType::NUM_OF_SCENE_TYPES);
    return static_cast<eSceneType>(i);
}

cSimScene::cSimScene()
{
    // mTriangleArray.clear();
    // mEdgeArray.clear();
    // mVertexArray.clear();

    mPerturb = nullptr;
    mPauseSim = false;
    // mColDetecter = nullptr;
    // mClothInitPos.setZero();
}

void cSimScene::Init(const std::string &conf_path)
{
    // 1. load config
    Json::Value root;
    cJsonUtil::LoadJson(conf_path, root);

    mEnableProfiling =
        cJsonUtil::ParseAsBool(cSimScene::ENABLE_PROFLINE_KEY, root);
    // mIdealDefaultTimestep =
    //     cJsonUtil::ParseAsDouble(cSimScene::DEFAULT_TIMESTEP_KEY, root);
    // mSceneType = BuildSceneType(
    //     cJsonUtil::ParseAsString(cSimScene::SCENE_TYPE_KEY, root));
    mEnableObstacle =
        cJsonUtil::ParseAsBool(cSimScene::ENABLE_OBSTACLE_KEY, root);

    mEnableCollisionDetection =
        cJsonUtil::ParseAsBool(cSimScene::ENABLE_COLLISION_DETECTION_KEY, root);

    CreateCloth(root);

    if (mEnableObstacle)
        CreateObstacle(
            cJsonUtil::ParseAsValue(cSimScene::OBSTACLE_CONF_KEY, root));
    if (mEnableCollisionDetection)
        CreateCollisionDetecter();

    InitDrawBuffer();
    InitRaycaster();
}
#include "sim/cloth/ClothBuilder.h"

void cSimScene::CreateCloth(const Json::Value &conf)
{
    // mCloth = std::make_shared<cSemiCloth>();
    mCloth = BuildCloth(conf);
    mCloth->Init(conf);
}

void cSimScene::PauseSim() { mPauseSim = !mPauseSim; }

void cSimScene::InitDrawBuffer()
{
    // 2. build arrays
    // init the buffer
    {
        int num_of_triangles = GetNumOfTriangles();
        // 1. add cloth triangles

        int num_of_vertices = num_of_triangles * 3;
        int size_per_vertices = RENDERING_SIZE_PER_VERTICE;
        int cloth_trinalge_size = num_of_vertices * size_per_vertices;

        mTriangleDrawBuffer.resize(cloth_trinalge_size);
        // std::cout << "triangle draw buffer size = " <<
        // mTriangleDrawBuffer.size() << std::endl; exit(0);
    }
    {

        int size_per_edge = 2 * RENDERING_SIZE_PER_VERTICE;
        mEdgesDrawBuffer.resize(GetNumOfEdges() * size_per_edge);
    }

    UpdateRenderingResource();
}

/**
 * \brief           Init the raycasting strucutre
 */
#include "geometries/OptixRaycaster.h"
#include "geometries/Raycaster.h"
void cSimScene::InitRaycaster()
{
    // auto total_triangle_array = mTriangleArray;
    // auto total_vertex_array = mVertexArray;
    // std::cout << "begin to add obstacle data array\n";
    // for (auto &x : mObstacleList)
    // {
    //     auto obstacle_v_array =  x->GetVertexArray();
    //     auto obstacle_triangle_array =  x->GetVertexArray();

    // }
    // for (int i = 0; i < this->)
#ifdef USE_OPTIX
    mRaycaster = std::make_shared<cOptixRaycaster>(false);
#else
    mRaycaster = std::make_shared<cRaycaster>(false);
#endif
    mRaycaster->AddResources(mCloth);
    for (auto &x : mObstacleList)
    {
        // auto obstacle_v_array = x->GetVertexArray();
        // auto obstacle_triangle_array = x->GetTriangleArray();
        mRaycaster->AddResources(x);
    }
    std::cout << "[debug] add resources to raycaster done, num of obstacles = "
              << mObstacleList.size() << std::endl;
}
/**
 * \brief           Update the simulation procedure
 */
#include "utils/TimeUtil.hpp"
void cSimScene::Update(double delta_time)
{
    // double default_dt = mIdealDefaultTimestep;
    // if (delta_time < default_dt)
    //     default_dt = delta_time;
    // printf("[debug] sim scene update cur time = %.4f\n", mCurTime);
    cScene::Update(delta_time);

    double dt = mCloth->GetDefaultTimestep();
    while (delta_time > 0)
    {
        delta_time -= dt;
        // 1. clera force
        mCloth->ClearForce();

        mCloth->ApplyPerturb(mPerturb);

        mCloth->UpdatePos(dt);
    }
    // mCloth->Update(delta_time);

    // clear force
    // apply ext force
    // update position
}

/**
 * \brief           Reset the whole scene
 */
void cSimScene::Reset()
{
    cScene::Reset();
    ClearForce();
}

/**
 * \brief           Get number of vertices
 */
int cSimScene::GetNumOfVertices() const
{
    // 1. get cloth vertices
    int num_of_vertices = 0;
    if (mCloth)
    {
        num_of_vertices += mCloth->GetNumOfVertices();
    }
    for (auto &x : mObstacleList)
    {
        num_of_vertices += x->GetNumOfVertices();
    }
    return num_of_vertices;
}

/**
 * \brief       clear all forces
 */
void cSimScene::ClearForce() { mCloth->ClearForce(); }

void cSimScene::GetVertexRenderingData() {}

int cSimScene::GetNumOfFreedom() const { return GetNumOfVertices() * 3; }

int cSimScene::GetNumOfEdges() const
{
    int num_of_edges = 0;
    if (mCloth)
    {
        num_of_edges += mCloth->GetNumOfEdges();
    }
    for (auto &x : mObstacleList)
    {
        num_of_edges += x->GetNumOfEdges();
    }
    return num_of_edges;
}

int cSimScene::GetNumOfTriangles() const
{
    int num_of_triangles = 0;
    if (mCloth)

    {
        num_of_triangles += mCloth->GetNumOfTriangles();
    }
    for (auto &x : mObstacleList)
    {
        num_of_triangles += x->GetNumOfTriangles();
    }
    return num_of_triangles;
}
/**
 * \brief       external force
 */
extern const tVector gGravity;

const tVectorXf &cSimScene::GetTriangleDrawBuffer()
{
    return mTriangleDrawBuffer;
}
/**
 * \brief           Calculate vertex rendering data
 */
void cSimScene::CalcTriangleDrawBuffer()
{
    mTriangleDrawBuffer.fill(std::nan(""));
    // 1. calculate for cloth triangle
    int st = 0;
    Eigen::Map<tVectorXf> ref(mTriangleDrawBuffer.data(),
                              mTriangleDrawBuffer.size());
    {
        mCloth->CalcTriangleDrawBuffer(ref, st);
    }
    // 2. calculate for obstacle triangle
    {
        for (auto &x : mObstacleList)
        {
            x->CalcTriangleDrawBuffer(ref, st);
        }
    }
}

const tVectorXf &cSimScene::GetEdgesDrawBuffer() { return mEdgesDrawBuffer; }

void cSimScene::UpdateRenderingResource()
{
    CalcEdgesDrawBuffer();
    CalcTriangleDrawBuffer();
}

void cSimScene::CalcEdgesDrawBuffer()
{
    mEdgesDrawBuffer.fill(std::nan(""));
    int st = 0;
    // 1. for cloth draw buffer
    Eigen::Map<tVectorXf> cloth_ref(mEdgesDrawBuffer.data(),
                                    mEdgesDrawBuffer.size());
    mCloth->CalcEdgeDrawBuffer(cloth_ref, st);

    // 2. for draw buffer
    for (auto &x : mObstacleList)
    {
        x->CalcEdgeDrawBuffer(cloth_ref, st);
    }
}

cSimScene::~cSimScene()
{
    // for (auto x : mVertexArray)
    //     delete x;
    // mVertexArray.clear();
    // for (auto &x : mTriangleArray)
    //     delete x;
    // mTriangleArray.clear();
    // for (auto &x : mEdgeArray)
    //     delete x;
    // mEdgeArray.clear();
}

/**
 * \brief               Event response (add perturb)
 */
void cSimScene::CursorMove(int xpos, int ypos) {}

void cSimScene::UpdatePerturb(const tVector &camera_pos, const tVector &dir)
{
    if (mPerturb != nullptr)
    {
        mPerturb->UpdatePerturb(camera_pos, dir);
    }
}
/**
 * \brief               Event response (add perturb)
 */
#include "scenes/DrawScene.h"
void cSimScene::MouseButton(int button, int action, int mods) {}
#include "GLFW/glfw3.h"
void cSimScene::Key(int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_I && action == GLFW_PRESS)
    {
        PauseSim();
    }
}

bool cSimScene::CreatePerturb(tRay *ray)
{
    // std::cout << "[warn] create perturb is disbled temprorarily\n";
    // return false;
    // 1. select triangle
    // tTriangle *selected_tri = nullptr;
    // tVector raycast_point = tVector::Zero();
    // int selected_tri_id = -1;
    // double min_depth = std::numeric_limits<double>::max();
    SIM_ASSERT(mRaycaster != nullptr);

    cRaycaster::tRaycastResult res = mRaycaster->RayCast(ray);
    if (res.mObject == nullptr)
        return false;
    else
    {
        std::cout << "[debug] add perturb on triangle " << res.mLocalTriangleId
                  << std::endl;
    }

    // 2. we have a triangle to track
    SIM_ASSERT(mPerturb == nullptr);

    mPerturb = new tPerturb();

    mPerturb->mObject = res.mObject;
    mPerturb->mAffectedTriId = res.mLocalTriangleId;
    const auto &ver_array = mPerturb->mObject->GetVertexArray();
    const auto &tri_array = mPerturb->mObject->GetTriangleArray();

    mPerturb->mBarycentricCoords =
        cMathUtil::CalcBarycentric(
            res.mIntersectionPoint,
            ver_array[tri_array[res.mLocalTriangleId]->mId0]->mPos,
            ver_array[tri_array[res.mLocalTriangleId]->mId1]->mPos,
            ver_array[tri_array[res.mLocalTriangleId]->mId2]->mPos)
            .segment(0, 3);
    SIM_ASSERT(mPerturb->mBarycentricCoords.hasNaN() == false);
    mPerturb->InitTangentRect(-1 * ray->mDir);
    mPerturb->UpdatePerturb(ray->mOrigin, ray->mDir);

    // // change the color
    mPerturb->mObject->ChangeTriangleColor(res.mLocalTriangleId,
                                           tVector(1, 0, 0, 0));
    // mVertexArray[selected_tri->mId0]->mColor = tVector(1, 0, 0, 0);
    // mVertexArray[selected_tri->mId1]->mColor = tVector(1, 0, 0, 0);
    // mVertexArray[selected_tri->mId2]->mColor = tVector(1, 0, 0, 0);
    // return true;
}

void cSimScene::ReleasePerturb()
{
    if (mPerturb != nullptr)
    {
        // restore the color
        mPerturb->mObject->ChangeTriangleColor(mPerturb->mAffectedTriId,
                                               tVector(0, 196.0 / 255, 1, 0));
        // mPerturb->mAffectedVertices[0]->mColor = ;
        // mPerturb->mAffectedVertices[1]->mColor = tVector(0, 196.0 / 255, 1,
        // 0); mPerturb->mAffectedVertices[2]->mColor = tVector(0, 196.0 / 255,
        // 1, 0);
        delete mPerturb;
        mPerturb = nullptr;
    }
}

void cSimScene::CreateObstacle(const Json::Value &conf)
{
    // 1. parse the number of obstacles
    Json::Value obstacles_lst = conf;
    int num_of_obstacles = obstacles_lst.size();
    SIM_ASSERT(num_of_obstacles == obstacles_lst.size());
    for (int i = 0; i < num_of_obstacles; i++)
    {
        auto obs = std::make_shared<cKinematicBody>();
        obs->Init(obstacles_lst[i]);
        mObstacleList.push_back(obs);
    }

    printf("[debug] create %d obstacle(s) done\n", mObstacleList.size());
    // exit(0);
}

// /**
//  * \brief                   Raycast the whole scene
//  * @param ray:              the given ray
//  * @param selected_tri:     a reference to selected triangle pointer
//  * @param selected_tri_id:  a reference to selected triangle id
//  * @param raycast_point:    a reference to intersection point
//  */
// void cSimScene::RayCastScene(const tRay *ray, cBaseObjectPtr casted_obj,
//                              int obj_triangle_id, tVector inter_point) const
// {
//     SIM_ASSERT(mRaycaster != nullptr);
//     cRaycaster::tRaycastResult res = mRaycaster->RayCast(ray);
//     casted_obj = res.mObject;
//     obj_triangle_id
// }

/**
 * \brief                   Collision Detection
 */
void cSimScene::CreateCollisionDetecter()
{
    // mColDetecter = std::make_shared<cCollisionDetecter>();
}

// int cSimScene::GetNumOfTriangles() const {
//     int num_of_tris_cloth = mCloth->GetNumOfTriangles();
//     int num_of_tris_obstacles = 0;
//     for(auto & x : mObstacleList){
//         num_of_tris_obstacles += x->GetNumOfTriangles();
//     }
// }