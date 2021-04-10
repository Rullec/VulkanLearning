#include "SimScene.h"
#include "geometries/Primitives.h"
#include "geometries/Triangulator.h"
#include "scenes/DrawScene.h"
#include "utils/JsonUtil.h"
#include <iostream>

std::string
    gIntegrationSchemeStr[eIntegrationScheme::NUM_OF_INTEGRATION_SCHEMES] = {
        "semi_implicit",          "implicit",  "opt_implicit", "tri_pbd",
        "tri_projective_dynamic", "tri_baraff"};

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
    // mClothInitPos.setZero();
}

void cSimScene::Init(const std::string &conf_path)
{
    // 1. load config
    Json::Value root;
    cJsonUtil::LoadJson(conf_path, root);

    mDamping = cJsonUtil::ParseAsDouble("damping", root);

    mIdealDefaultTimestep = cJsonUtil::ParseAsDouble("default_timestep", root);
    mScheme = BuildIntegrationScheme(
        cJsonUtil::ParseAsString("integration_scheme", root));

    std::cout << "init sim scene done\n";
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
}

void cSimScene::UpdateCurNodalPosition(const tVectorXd &newpos)
{
    mXcur = newpos;
    for (int i = 0; i < mVertexArray.size(); i++)
    {
        mVertexArray[i]->mPos.segment(0, 3).noalias() = mXcur.segment(i * 3, 3);
    }
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

/**
 * \brief       external force
*/
extern const tVector gGravity;
void cSimScene::CalcExtForce(tVectorXd &ext_force) const
{
// apply gravity
#pragma omp parallel for
    for (int i = 0; i < mVertexArray.size(); i++)
    {
        ext_force.segment(3 * i, 3) +=
            gGravity.segment(0, 3) * mVertexArray[i]->mMass;
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
void cSimScene::CursorMove(cDrawScene *draw_scene, int xpos, int ypos) {}

/**
 * \brief               Event response (add perturb)
*/
void cSimScene::MouseButton(cDrawScene *draw_scene, int button, int action,
                            int mods)
{
    // std::cout << "[sim] mouse button\n";
    if (cDrawScene::IsMouseRightButton(button) == true)
    {
        // std::cout << "[sim] mouse button right\n";
        if (cDrawScene::IsPress(action) == true)
        {
            // std::cout << "[sim] mouse button right press\n";
            tVector tar_pos = draw_scene->CalcCursorPointWorldPos();
            tVector camera_pos = draw_scene->GetCameraPos();
            tRay *ray = new tRay(camera_pos, tar_pos);
            RayCast(ray);
        }
        else if (cDrawScene::IsRelease(action) == true)
        {
        }
    }
}

void cSimScene::InitGeometry(const Json::Value &conf)
{
    cTriangulator::BuildGeometry(conf, mVertexArray, mEdgeArray,
                                 mTriangleArray);
}

void cSimScene::RayCast(tRay *ray)
{
    std::cout << "begin to do ray cast for ray from "
              << ray->mOrigin.transpose() << " to " << ray->mDir.transpose()
              << std::endl;
    // for ()
    for (int i = 0; i < mTriangleArray.size(); i++)
    {
        auto &tri = mTriangleArray[i];
        tVector res = cMathUtil::RayCast(
            ray->mOrigin, ray->mDir, mVertexArray[tri->mId0]->mPos,
            mVertexArray[tri->mId1]->mPos, mVertexArray[tri->mId2]->mPos);
        if (res.hasNaN() == false)
        {
            std::cout << "intersect with triangle " << i << std::endl;
            mVertexArray[tri->mId0]->mColor = tVector(1, 0, 0, 0);
            mVertexArray[tri->mId1]->mColor = tVector(1, 0, 0, 0);
            mVertexArray[tri->mId2]->mColor = tVector(1, 0, 0, 0);
        }
    }

    mRayArray.push_back(ray);
}