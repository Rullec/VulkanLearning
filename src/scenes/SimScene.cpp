#include "SimScene.h"
#include "Perturb.h"
#include "geometries/Primitives.h"
#include "geometries/Triangulator.h"
#include "sim/CollisionDetecter.h"
#include "sim/KinematicBody.h"
#include "utils/JsonUtil.h"
#include <iostream>

std::string gSceneTypeStr[eSceneType::NUM_OF_SCENE_TYPES] = {
    "semi_implicit", "implicit", "projective_dynamic", "pbd",
    "tri_baraff",    "se",       "data_synthesis",     "data_process",
    "mesh_vis"};

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
    mTriangleArray.clear();
    mEdgeArray.clear();
    mVertexArray.clear();
    mFixedPointIds.clear();
    mPerturb = nullptr;
    mPauseSim = false;
    mColDetecter = nullptr;
    // mClothInitPos.setZero();
}

void cSimScene::Init(const std::string &conf_path)
{
    // 1. load config
    Json::Value root;
    cJsonUtil::LoadJson(conf_path, root);

    mGeometryType =
        cJsonUtil::ParseAsString(cTriangulator::GEOMETRY_TYPE_KEY, root);
    mDamping = cJsonUtil::ParseAsDouble(cSimScene::DAMPING_KEY, root);
    mEnableProfiling =
        cJsonUtil::ParseAsBool(cSimScene::ENABLE_PROFLINE_KEY, root);
    mIdealDefaultTimestep =
        cJsonUtil::ParseAsDouble(cSimScene::DEFAULT_TIMESTEP_KEY, root);
    mSceneType = BuildSceneType(
        cJsonUtil::ParseAsString(cSimScene::SCENE_TYPE_KEY, root));
    mEnableObstacle =
        cJsonUtil::ParseAsBool(cSimScene::ENABLE_OBSTACLE_KEY, root);

    mEnableCollisionDetection =
        cJsonUtil::ParseAsBool(cSimScene::ENABLE_COLLISION_DETECTION_KEY, root);
    if (mEnableObstacle)
        CreateObstacle(
            cJsonUtil::ParseAsValue(cSimScene::OBSTACLE_CONF_KEY, root));
    if (mEnableCollisionDetection)
        CreateCollisionDetecter();
}

void cSimScene::PauseSim() { mPauseSim = !mPauseSim; }

void cSimScene::InitDrawBuffer()
{
    // 2. build arrays
    // init the buffer
    {
        int num_of_triangles_cloth = mTriangleArray.size();
        int num_of_triangles_obstacle = 0;
        for (auto &x : mObstacleList)
        {
            num_of_triangles_obstacle += x->GetDrawNumOfTriangles();
        }
        int num_of_triangles =
            num_of_triangles_cloth + num_of_triangles_obstacle;
        int num_of_vertices = num_of_triangles * 3;
        int size_per_vertices = RENDERING_SIZE_PER_VERTICE;
        int cloth_trinalge_size = num_of_vertices * size_per_vertices;

        mTriangleDrawBuffer.resize(cloth_trinalge_size);
        // std::cout << "triangle draw buffer size = " <<
        // mTriangleDrawBuffer.size() << std::endl; exit(0);
    }
    {
        int num_of_edges_cloth = mEdgeArray.size();
        int num_of_edges_obstacle = 0;
        for (auto &x : mObstacleList)
        {
            num_of_edges_obstacle += x->GetDrawNumOfEdges();
        }
        int num_of_edges = num_of_edges_obstacle + num_of_edges_cloth;
        int size_per_edge = 2 * RENDERING_SIZE_PER_VERTICE;
        mEdgesDrawBuffer.resize(num_of_edges * size_per_edge);
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
    mRaycaster->AddResources(mTriangleArray, mVertexArray);
    for (auto &x : mObstacleList)
    {
        auto obstacle_v_array = x->GetVertexArray();
        auto obstacle_triangle_array = x->GetTriangleArray();
        mRaycaster->AddResources(obstacle_triangle_array, obstacle_v_array);
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
    double default_dt = mIdealDefaultTimestep;
    // if (delta_time < default_dt)
    //     default_dt = delta_time;
    // printf("[debug] sim scene update cur time = %.4f\n", mCurTime);
    while (delta_time > 1e-7)
    {
        // if (delta_time < default_dt)
        //     default_dt = delta_time;
        cScene::Update(default_dt);

        // cTimeUtil::Begin("substep");
        UpdateSubstep();
        // cTimeUtil::End("substep");

        delta_time -= default_dt;
    }

    // 4. post process
    // CalcTriangleDrawBuffer();
    // CalcEdgesDrawBuffer();
    // std::cout << "xcur = " << mXcur.transpose() << std::endl;
    SIM_ASSERT(mXcur.hasNaN() == false);
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
int cSimScene::GetNumOfVertices() const { return mVertexArray.size(); }

/**
 * \brief       clear all forces
 */
void cSimScene::ClearForce()
{
    int dof = GetNumOfFreedom();
    mIntForce.noalias() = tVectorXd::Zero(dof);
    mExtForce.noalias() = tVectorXd::Zero(dof);
    mDampingForce.noalias() = tVectorXd::Zero(dof);
}

void cSimScene::UpdateCurNodalPosition(const tVectorXd &newpos)
{
    mXcur = newpos;
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < mVertexArray.size(); i++)
    {
        mVertexArray[i]->mPos.segment(0, 3).noalias() = mXcur.segment(i * 3, 3);
    }
}
/**
 * \brief           add damping forces
 */
void cSimScene::CalcDampingForce(const tVectorXd &vel, tVectorXd &damping) const
{
    damping.noalias() = -vel * mDamping;
}

void cSimScene::GetVertexRenderingData() {}

int cSimScene::GetNumOfFreedom() const { return GetNumOfVertices() * 3; }

void CalcTriangleDrawBufferSingle(tVertex *v0, tVertex *v1, tVertex *v2,
                                  Eigen::Map<tVectorXf> &buffer, int &st_pos)
{
    // std::cout << "buffer size " << buffer.size() << " st pos " << st_pos <<
    // std::endl;
    buffer.segment(st_pos, 3) = v0->mPos.segment(0, 3).cast<float>();
    buffer.segment(st_pos + 3, 3) = v0->mColor.segment(0, 3).cast<float>();
    st_pos += RENDERING_SIZE_PER_VERTICE;
    buffer.segment(st_pos, 3) = v1->mPos.segment(0, 3).cast<float>();
    buffer.segment(st_pos + 3, 3) = v1->mColor.segment(0, 3).cast<float>();
    st_pos += RENDERING_SIZE_PER_VERTICE;
    buffer.segment(st_pos, 3) = v2->mPos.segment(0, 3).cast<float>();
    buffer.segment(st_pos + 3, 3) = v2->mColor.segment(0, 3).cast<float>();
    st_pos += RENDERING_SIZE_PER_VERTICE;
}
int cSimScene::GetNumOfEdges() const { return mEdgeArray.size(); }
/**
 * \brief       external force
 */
extern const tVector gGravity;
void cSimScene::CalcExtForce(tVectorXd &ext_force) const
{
// 1. apply gravity
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < mVertexArray.size(); i++)
    {
        ext_force.segment(3 * i, 3) +=
            gGravity.segment(0, 3) * mVertexArray[i]->mMass;
    }

    // std::cout << "add ext noise\n";
    // ext_force.segment(3 * (mVertexArray.size() - 1), 3) += tVector3d(0, 0,
    // 10);

    //  2. add perturb force
    if (mPerturb != nullptr)
    {
        tVector perturb_force = mPerturb->GetPerturbForce();
        // printf(
        //     "[debug] perturb vid %d %d %d, ",
        //     mPerturb->mAffectedVerticesId[0],
        //     mPerturb->mAffectedVerticesId[1], mPerturb        // std::cout <<
        //     "perturb force = " << perturb_force.transpose()
        //           << std::endl;->mAffectedVerticesId[2]);

        ext_force.segment(mPerturb->mAffectedVerticesId[0] * 3, 3) +=
            perturb_force.segment(0, 3) / 3;
        ext_force.segment(mPerturb->mAffectedVerticesId[1] * 3, 3) +=
            perturb_force.segment(0, 3) / 3;
        ext_force.segment(mPerturb->mAffectedVerticesId[2] * 3, 3) +=
            perturb_force.segment(0, 3) / 3;
        // 2. give the ray to the perturb, calculate force on each vertices
        // 3. apply the force
    }

    std::cout << "[debug] add collision spring force\n";
    if (mColDetecter != nullptr)
    {
        auto pts = mColDetecter->GetCollisionPoints();
        for (auto &pt : pts)
        {
            double pene = -1 * (pt->mPenetration - 0.005);
            double k = 1e3;
            pene = pene < 0 ? 0 : pene;
            double force = k * pene;
            ext_force.segment(pt->mVertexId * 3, 3) +=
                pt->mContactNormal.segment(0, 3) * force;
            std::cout << "[debug] col force " << force << " on vertex "
                      << pt->mVertexId << std::endl;
        }
    }
}

void CalcEdgeDrawBufferSingle(tVertex *v0, tVertex *v1,
                              Eigen::Map<tVectorXf> &buffer, int &st_pos)
{

    buffer.segment(st_pos, 3) = v0->mPos.segment(0, 3).cast<float>();
    buffer.segment(st_pos + 3, 3) = tVector3f(0, 0, 0);
    st_pos += RENDERING_SIZE_PER_VERTICE;
    buffer.segment(st_pos, 3) = v1->mPos.segment(0, 3).cast<float>();
    buffer.segment(st_pos + 3, 3) = tVector3f(0, 0, 0);
    st_pos += RENDERING_SIZE_PER_VERTICE;
}

void CalcEdgeDrawBufferSingle(const tVector &v0, const tVector &v1,
                              Eigen::Map<tVectorXf> &buffer, int &st_pos)
{
    buffer.segment(st_pos, 3) = v0.segment(0, 3).cast<float>();
    buffer.segment(st_pos + 3, 3) = tVector3f(0, 0, 0);
    st_pos += RENDERING_SIZE_PER_VERTICE;
    buffer.segment(st_pos, 3) = v1.segment(0, 3).cast<float>();
    buffer.segment(st_pos + 3, 3) = tVector3f(1, 0, 0);
    st_pos += RENDERING_SIZE_PER_VERTICE;
}
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
    {
        Eigen::Map<tVectorXf> ref(mTriangleDrawBuffer.data(),
                                  mTriangleDrawBuffer.size());
        // counter clockwise
        int subdivision = std::sqrt(mVertexArray.size()) - 1;
        int gap = subdivision + 1;
        for (int i = 0; i < subdivision; i++)     // row
            for (int j = 0; j < subdivision; j++) // column
            {
                // left up coner
                int left_up = gap * i + j;
                int right_up = left_up + 1;
                int left_down = left_up + gap;
                int right_down = right_up + gap;
                // mVertexArray[left_up]->mPos *= (1 + 1e-3);
                CalcTriangleDrawBufferSingle(mVertexArray[right_down],
                                             mVertexArray[left_up],
                                             mVertexArray[left_down], ref, st);
                CalcTriangleDrawBufferSingle(mVertexArray[right_down],
                                             mVertexArray[right_up],
                                             mVertexArray[left_up], ref, st);
            }
    }
    // 2. calculate for obstacle triangle
    {
        for (auto &x : mObstacleList)
        {
            Eigen::Map<tVectorXf> ref(mTriangleDrawBuffer.data() + st,
                                      mTriangleDrawBuffer.size() - st);
            x->CalcTriangleDrawBuffer(ref);
            st += x->GetDrawNumOfTriangles() * 3 * RENDERING_SIZE_PER_VERTICE;
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
    for (auto &e : mEdgeArray)
    {
        CalcEdgeDrawBufferSingle(mVertexArray[e->mId0], mVertexArray[e->mId1],
                                 cloth_ref, st);
    }
    // 2. for draw buffer
    {
        if (mObstacleList.empty() == false)
        {
            // std::cout << "[debug] calc edge draw buffer obstacle, size = " <<
            // ref.size() << std::endl;
            for (auto &x : mObstacleList)
            {
                int size =
                    x->GetDrawNumOfEdges() * RENDERING_SIZE_PER_VERTICE * 2;
                Eigen::Map<tVectorXf> ref(mEdgesDrawBuffer.data() + st,
                                          mEdgesDrawBuffer.size() - st);
                x->CalcEdgeDrawBuffer(ref);
                st += size;
            }
        }
    }
}

void cSimScene::CalcNodePositionVector(tVectorXd &pos) const
{
    if (pos.size() != GetNumOfFreedom())
    {
        pos.noalias() = tVectorXd::Zero(GetNumOfFreedom());
    }
    for (int i = 0; i < mVertexArray.size(); i++)
    {
        pos.segment(i * 3, 3) = mVertexArray[i]->mPos.segment(0, 3);
    }
}

void cSimScene::InitConstraint(const Json::Value &root)
{
    if (root.isMember("constraints") == false)
        return;
    auto cons = cJsonUtil::ParseAsValue("constraints", root);
    if (cons.isMember("fixed_point") == true)
    {
        // 1. read all 2d constraint for fixed point
        auto fixed_cons = cJsonUtil::ParseAsValue("fixed_point", cons);
        int num_of_fixed_pts = fixed_cons.size();
        tEigenArr<tVector2f> fixed_tex_coords(num_of_fixed_pts);
        for (int i = 0; i < num_of_fixed_pts; i++)
        {
            SIM_ASSERT(fixed_cons[i].size() == 2);
            fixed_tex_coords[i] = tVector2f(fixed_cons[i][0].asDouble(),
                                            fixed_cons[i][1].asDouble());
        }

        // 2. iterate over all vertices to find which point should be finally
        // fixed
        mFixedPointIds.resize(num_of_fixed_pts, -1);
        std::vector<double> SelectedFixedPointApproxDist(num_of_fixed_pts,
                                                         std::nan(""));

        for (int v_id = 0; v_id < GetNumOfVertices(); v_id++)
        {
            const tVector2f &v_uv = mVertexArray[v_id]->muv;
            for (int j = 0; j < num_of_fixed_pts; j++)
            {
                double dist = (v_uv - fixed_tex_coords[j]).norm();
                if (std::isnan(SelectedFixedPointApproxDist[j]) ||
                    dist < SelectedFixedPointApproxDist[j])
                {
                    mFixedPointIds[j] = v_id;
                    SelectedFixedPointApproxDist[j] = dist;
                }
            }
        }

        // output
        for (int i = 0; i < num_of_fixed_pts; i++)
        {
            printf("[debug] fixed uv (%.3f %.3f) selected v_id %d uv (%.3f, "
                   "%.3f)\n",
                   fixed_tex_coords[i][0], fixed_tex_coords[i][1],
                   mFixedPointIds[i], mVertexArray[mFixedPointIds[i]]->muv[0],
                   mVertexArray[mFixedPointIds[i]]->muv[1]);
        }
    }
    for (auto &i : mFixedPointIds)
    {
        mInvMassMatrixDiag.segment(i * 3, 3).setZero();
        // printf("[debug] fixed point id %d at ", i);
        // exit(0);
        // next_pos.segment(i * 3, 3) = mXcur.segment(i * 3, 3);
        // std::cout << mXcur.segment(i * 3, 3).transpose() << std::endl;
    }
}

cSimScene::~cSimScene()
{
    for (auto x : mVertexArray)
        delete x;
    mVertexArray.clear();
    for (auto &x : mTriangleArray)
        delete x;
    mTriangleArray.clear();
    for (auto &x : mEdgeArray)
        delete x;
    mEdgeArray.clear();
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
void cSimScene::MouseButton(int button, int action, int mods)
{
    // if (cDrawScene::IsMouseRightButton(button) == true)
    // {
    //     if (cDrawScene::IsPress(action) == true)
    //     {
    //         tVector tar_pos = draw_scene->CalcCursorPointWorldPos();
    //         tVector camera_pos = draw_scene->GetCameraPos();
    //         tRay *ray = new tRay(camera_pos, tar_pos);
    //         CreatePerturb(ray);
    //     }
    //     else if (cDrawScene::IsRelease(action) == true)
    //     {

    //         ReleasePerturb();
    //     }
    // }
}
#include "GLFW/glfw3.h"
void cSimScene::Key(int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_I && action == GLFW_PRESS)
    {
        PauseSim();
    }
}
void cSimScene::InitGeometry(const Json::Value &conf)
{
    // 1. build the geometry
    mClothWidth = cJsonUtil::ParseAsDouble("cloth_size", conf);
    mClothMass = cJsonUtil::ParseAsDouble("cloth_mass", conf);

    cTriangulator::BuildGeometry(conf, mVertexArray, mEdgeArray,
                                 mTriangleArray);

    CalcNodePositionVector(mClothInitPos);

    // init the inv mass vector
    mInvMassMatrixDiag.noalias() = tVectorXd::Zero(GetNumOfFreedom());
    for (int i = 0; i < mVertexArray.size(); i++)
    {
        mInvMassMatrixDiag.segment(i * 3, 3).fill(1.0 / mVertexArray[i]->mMass);
    }
}

bool cSimScene::CreatePerturb(tRay *ray)
{
    // std::cout << "begin to do ray cast for ray from "
    //           << ray->mOrigin.transpose() << " to " << ray->mDir.transpose()
    //           << std::endl;
    // 1. select triangle
    tTriangle *selected_tri = nullptr;
    tVector raycast_point = tVector::Zero();
    int selected_tri_id = -1;
    // double min_depth = std::numeric_limits<double>::max();
    RayCastScene(ray, &selected_tri, selected_tri_id, raycast_point);
    if (selected_tri == nullptr)
        return false;
    else
    {
        std::cout << "[debug] add perturb on triangle " << selected_tri_id
                  << std::endl;
    }

    // 2. we have a triangle to track
    SIM_ASSERT(mPerturb == nullptr);

    mPerturb = new tPerturb();

    mPerturb->mAffectedTriId = selected_tri_id;
    mPerturb->mAffectedVerticesId[0] = selected_tri->mId0;
    mPerturb->mAffectedVerticesId[1] = selected_tri->mId1;
    mPerturb->mAffectedVerticesId[2] = selected_tri->mId2;

    mPerturb->mAffectedVertices[0] = mVertexArray[selected_tri->mId0];
    mPerturb->mAffectedVertices[1] = mVertexArray[selected_tri->mId1];
    mPerturb->mAffectedVertices[2] = mVertexArray[selected_tri->mId2];
    mPerturb->mBarycentricCoords =
        cMathUtil::CalcBarycentric(raycast_point,
                                   mVertexArray[selected_tri->mId0]->mPos,
                                   mVertexArray[selected_tri->mId1]->mPos,
                                   mVertexArray[selected_tri->mId2]->mPos)
            .segment(0, 3);
    SIM_ASSERT(mPerturb->mBarycentricCoords.hasNaN() == false);
    mPerturb->InitTangentRect(-1 * ray->mDir);
    mPerturb->UpdatePerturb(ray->mOrigin, ray->mDir);

    // change the color
    mVertexArray[selected_tri->mId0]->mColor = tVector(1, 0, 0, 0);
    mVertexArray[selected_tri->mId1]->mColor = tVector(1, 0, 0, 0);
    mVertexArray[selected_tri->mId2]->mColor = tVector(1, 0, 0, 0);
    return true;
}

/**
 * \brief       internal force
 */
void cSimScene::CalcIntForce(const tVectorXd &xcur, tVectorXd &int_force) const
{
    // std::vector<std::atomic<double>> int_force_atomic(int_force.size());
    // for (int i = 0; i < int_force.size(); i++)
    //     int_force_atomic[i] = 0;
    // double res = 1;
    // std::vector<double> int_force_atomic(int_force.size());

    // std::cout << "input fint = " << int_force.transpose() << std::endl;
    int id0, id1;
    double dist;
#ifdef USE_OPENMP
#pragma omp parallel for private(id0, id1, dist)
#endif
    for (int i = 0; i < mEdgeArray.size(); i++)
    {
        const auto &spr = mEdgeArray[i];
        // 1. calcualte internal force for each spring
        id0 = spr->mId0;
        id1 = spr->mId1;
        tVector3d pos0 = xcur.segment(id0 * 3, 3);
        tVector3d pos1 = xcur.segment(id1 * 3, 3);
        dist = (pos0 - pos1).norm();
        tVector3d force0 = spr->mK_spring * (spr->mRawLength - dist) *
                           (pos0 - pos1).segment(0, 3) / dist;
        // tVector3d force1 = -force0;
        // const tVectorXd &inf_force_0 = int_force.segment(3 * id0, 3);
        // const tVectorXd &inf_force_1 = int_force.segment(3 * id1, 3);
        //         std::cout << "spring " << i << " force = " <<
        //         force0.transpose() << ", dist " << dist << ", v0 " << id0 <<
        //         " v1 " << id1 << std::endl;
        // std::cout << "spring " << i << ", v0 = " << id0 << " v1 = " << id1 <<
        // std::endl;
        // 2. add force
        {
#ifdef USE_OPENMP
#pragma omp atomic
#endif
            int_force[3 * id0 + 0] += force0[0];
#ifdef USE_OPENMP
#pragma omp atomic
#endif
            int_force[3 * id0 + 1] += force0[1];
#ifdef USE_OPENMP
#pragma omp atomic
#endif
            int_force[3 * id0 + 2] += force0[2];
#ifdef USE_OPENMP
#pragma omp atomic
#endif
            int_force[3 * id1 + 0] += -force0[0];
#ifdef USE_OPENMP
#pragma omp atomic
#endif
            int_force[3 * id1 + 1] += -force0[1];
#ifdef USE_OPENMP
#pragma omp atomic
#endif
            int_force[3 * id1 + 2] += -force0[2];
        }
    }
    // std::cout << "output fint = " << int_force.transpose() << std::endl;
    // exit(0);
}

void cSimScene::ReleasePerturb()
{
    if (mPerturb != nullptr)
    {
        // restore the color
        mPerturb->mAffectedVertices[0]->mColor = tVector(0, 196.0 / 255, 1, 0);
        mPerturb->mAffectedVertices[1]->mColor = tVector(0, 196.0 / 255, 1, 0);
        mPerturb->mAffectedVertices[2]->mColor = tVector(0, 196.0 / 255, 1, 0);
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

/**
 * \brief                   Raycast the whole scene
 * @param ray:              the given ray
 * @param selected_tri:     a reference to selected triangle pointer
 * @param selected_tri_id:  a reference to selected triangle id
 * @param raycast_point:    a reference to intersection point
 */
void cSimScene::RayCastScene(const tRay *ray, tTriangle **selected_tri,
                             int &selected_tri_id, tVector &raycast_point) const
{
    SIM_ASSERT(mRaycaster != nullptr);
    mRaycaster->RayCast(ray, selected_tri, selected_tri_id, raycast_point);
}

/**
 * \brief                   Collision Detection
 */
void cSimScene::CreateCollisionDetecter()
{
    mColDetecter = std::make_shared<cCollisionDetecter>();
}