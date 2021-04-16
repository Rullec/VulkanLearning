#include "SimScene.h"
#include "Perturb.h"
#include "geometries/Primitives.h"
#include "geometries/Triangulator.h"
#include "scenes/DrawScene.h"
#include "utils/JsonUtil.h"
#include <iostream>

std::string
    gIntegrationSchemeStr[eIntegrationScheme::NUM_OF_INTEGRATION_SCHEMES] = {
        "semi_implicit", "implicit", "projective_dynamic", "pbd",
        "tri_baraff", "se"};

eIntegrationScheme cSimScene::BuildIntegrationScheme(const std::string &str)
{
    int i = 0;
    for (i = 0; i < eIntegrationScheme::NUM_OF_INTEGRATION_SCHEMES; i++)
    {
        if (str == gIntegrationSchemeStr[i])
        {
            break;
        }
    }

    SIM_ASSERT(i != eIntegrationScheme::NUM_OF_INTEGRATION_SCHEMES);
    return static_cast<eIntegrationScheme>(i);
}

cSimScene::cSimScene()
{
    mTriangleArray.clear();
    mEdgeArray.clear();
    mVertexArray.clear();
    mFixedPointIds.clear();
    mPerturb = nullptr;
    // mClothInitPos.setZero();
}

void cSimScene::Init(const std::string &conf_path)
{
    // 1. load config
    Json::Value root;
    cJsonUtil::LoadJson(conf_path, root);

    mGeometryType = cJsonUtil::ParseAsString("geometry_type", root);
    mDamping = cJsonUtil::ParseAsDouble("damping", root);
    mEnableProfiling = cJsonUtil::ParseAsBool("enable_profiling", root);
    mIdealDefaultTimestep = cJsonUtil::ParseAsDouble("default_timestep", root);
    mScheme = BuildIntegrationScheme(
        cJsonUtil::ParseAsString("integration_scheme", root));
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
    CalcTriangleDrawBuffer();
    CalcEdgesDrawBuffer();
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
                                  tVectorXf &buffer, int &st_pos)
{
    // std::cout << "buffer size " << buffer.size() << " st pos " << st_pos << std::endl;
    buffer.segment(st_pos, 3) = v0->mPos.segment(0, 3).cast<float>();
    buffer.segment(st_pos + 3, 3) = v0->mColor.segment(0, 3).cast<float>();
    st_pos += 8;
    buffer.segment(st_pos, 3) = v1->mPos.segment(0, 3).cast<float>();
    buffer.segment(st_pos + 3, 3) = v1->mColor.segment(0, 3).cast<float>();
    st_pos += 8;
    buffer.segment(st_pos, 3) = v2->mPos.segment(0, 3).cast<float>();
    buffer.segment(st_pos + 3, 3) = v2->mColor.segment(0, 3).cast<float>();
    st_pos += 8;
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
    // ext_force.segment(3 * (mVertexArray.size() - 1), 3) += tVector3d(0, 0, 10);

    //  2. add perturb force
    if (mPerturb != nullptr)
    {
        tVector perturb_force = mPerturb->GetPerturbForce();
        // printf(
        //     "[debug] perturb vid %d %d %d, ", mPerturb->mAffectedVerticesId[0],
        //     mPerturb->mAffectedVerticesId[1], mPerturb        // std::cout << "perturb force = " << perturb_force.transpose()
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
}

void CalcEdgeDrawBufferSingle(tVertex *v0, tVertex *v1, tVectorXf &buffer,
                              int &st_pos)
{

    buffer.segment(st_pos, 3) = v0->mPos.segment(0, 3).cast<float>();
    buffer.segment(st_pos + 3, 3) = tVector3f(0, 0, 0);
    st_pos += 8;
    buffer.segment(st_pos, 3) = v1->mPos.segment(0, 3).cast<float>();
    buffer.segment(st_pos + 3, 3) = tVector3f(0, 0, 0);
    st_pos += 8;
}

void CalcEdgeDrawBufferSingle(const tVector &v0, const tVector &v1,
                              tVectorXf &buffer, int &st_pos)
{
    buffer.segment(st_pos, 3) = v0.segment(0, 3).cast<float>();
    buffer.segment(st_pos + 3, 3) = tVector3f(0, 0, 0);
    st_pos += 8;
    buffer.segment(st_pos, 3) = v1.segment(0, 3).cast<float>();
    buffer.segment(st_pos + 3, 3) = tVector3f(1, 0, 0);
    st_pos += 8;
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
    // counter clockwise
    int subdivision = std::sqrt(mVertexArray.size()) - 1;
    int gap = subdivision + 1;
    int st = 0;
    for (int i = 0; i < subdivision; i++)     // row
        for (int j = 0; j < subdivision; j++) // column
        {
            // left up coner
            int left_up = gap * i + j;
            int right_up = left_up + 1;
            int left_down = left_up + gap;
            int right_down = right_up + gap;
            // mVertexArray[left_up]->mPos *= (1 + 1e-3);
            CalcTriangleDrawBufferSingle(
                mVertexArray[right_down], mVertexArray[left_up],
                mVertexArray[left_down], mTriangleDrawBuffer, st);
            CalcTriangleDrawBufferSingle(
                mVertexArray[right_down], mVertexArray[right_up],
                mVertexArray[left_up], mTriangleDrawBuffer, st);
        }
}

const tVectorXf &cSimScene::GetEdgesDrawBuffer() { return mEdgesDrawBuffer; }

void cSimScene::CalcEdgesDrawBuffer()
{
    mEdgesDrawBuffer.fill(std::nan(""));
    int st = 0;
    for (auto &e : mEdgeArray)
    {
        CalcEdgeDrawBufferSingle(mVertexArray[e->mId0], mVertexArray[e->mId1],
                                 mEdgesDrawBuffer, st);
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

        // 2. iterate over all vertices to find which point should be finally fixed
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
void cSimScene::CursorMove(cDrawScene *draw_scene, int xpos, int ypos)
{
    if (mPerturb != nullptr)
    {
        // update perturb
        tVector camera_pos = draw_scene->GetCameraPos();
        tVector dir = draw_scene->CalcCursorPointWorldPos() - camera_pos;
        dir[3] = 0;
        dir.normalize();
        mPerturb->UpdatePerturb(camera_pos, dir);
        // std::cout << "now perturb force = "
        //           << mPerturb->GetPerturbForce().transpose() << std::endl;
    }
}

/**
 * \brief               Event response (add perturb)
*/
void cSimScene::MouseButton(cDrawScene *draw_scene, int button, int action,
                            int mods)
{
    if (cDrawScene::IsMouseRightButton(button) == true)
    {
        if (cDrawScene::IsPress(action) == true)
        {
            tVector tar_pos = draw_scene->CalcCursorPointWorldPos();
            tVector camera_pos = draw_scene->GetCameraPos();
            tRay *ray = new tRay(camera_pos, tar_pos);
            CreatePerturb(ray);
        }
        else if (cDrawScene::IsRelease(action) == true)
        {

            ReleasePerturb();
        }
    }
}

void cSimScene::InitGeometry(const Json::Value &conf)
{
    // 1. build the geometry
    cTriangulator::BuildGeometry(conf, mVertexArray, mEdgeArray,
                                 mTriangleArray);
    // 2. build arrays
    // init the buffer
    {
        int num_of_triangles = mTriangleArray.size();
        int num_of_vertices = num_of_triangles * 3;
        int size_per_vertices = 8;
        mTriangleDrawBuffer.resize(num_of_vertices * size_per_vertices);
        // std::cout << "triangle draw buffer size = " << mTriangleDrawBuffer.size() << std::endl;
        // exit(0);
    }
    {
        int num_of_edges = mEdgeArray.size();
        int size_per_edge = 16;
        mEdgesDrawBuffer.resize(num_of_edges * size_per_edge);
    }

    CalcTriangleDrawBuffer();
    CalcEdgesDrawBuffer();

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
    for (int i = 0; i < mTriangleArray.size(); i++)
    {
        auto &tri = mTriangleArray[i];
        raycast_point = cMathUtil::RayCast(
            ray->mOrigin, ray->mDir, mVertexArray[tri->mId0]->mPos,
            mVertexArray[tri->mId1]->mPos, mVertexArray[tri->mId2]->mPos);
        if (raycast_point.hasNaN() == false)
        {
            std::cout << "[debug] add perturb on triangle " << i << std::endl;
            selected_tri = tri;
            selected_tri_id = i;
            break;
            // mVertexArray[tri->mId0]->mColor = tVector(1, 0, 0, 0);
            // mVertexArray[tri->mId1]->mColor = tVector(1, 0, 0, 0);
            // mVertexArray[tri->mId2]->mColor = tVector(1, 0, 0, 0);
        }
    }
    if (selected_tri == nullptr)
        return false;

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
    // std::cout << "bary coords = " << mPerturb->mBarycentricCoords.transpose()
    //           << std::endl;
    mPerturb->InitTangentRect(-1 * ray->mDir);

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
        //         std::cout << "spring " << i << " force = " << force0.transpose() << ", dist " << dist << ", v0 " << id0 << " v1 " << id1 << std::endl;
        // std::cout << "spring " << i << ", v0 = " << id0 << " v1 = " << id1 << std::endl;
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
        mPerturb->mAffectedVertices[0]->mColor =
            tVector(0, 196.0 / 255, 1, 0);
        mPerturb->mAffectedVertices[1]->mColor =
            tVector(0, 196.0 / 255, 1, 0);
        mPerturb->mAffectedVertices[2]->mColor =
            tVector(0, 196.0 / 255, 1, 0);
        delete mPerturb;
        mPerturb = nullptr;
    }
}
