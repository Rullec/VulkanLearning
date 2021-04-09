#include "TrimeshScene.h"
#include <atomic>
#include <omp.h>
#include "utils/LogUtil.h"
#include "geometries/Primitives.h"
#include <set>
#include <iostream>
cTrimeshScene::cTrimeshScene()
{
    mTriangleArray.clear();
    mEdgeArray.clear();
    mVcur.resize(0);
}

cTrimeshScene::~cTrimeshScene()
{
    for (auto &x : mTriangleArray)
        delete x;
    for (auto &x : mEdgeArray)
        delete x;
    mTriangleArray.clear();
    mEdgeArray.clear();
}

/**
 * \brief               build trimesh
 * 1. build vertex
 * 2. build edge
 * 3. build triangles
 * 
 *  For more details, please check the note "将平面划分为三角形.md"
*/
#include "geometries/Triangulator.h"
void cTrimeshScene::InitGeometry(const Json::Value &conf)
{

    cTriangulator::BuildGeometry(
        conf,
        mVertexArray, mEdgeArray, mTriangleArray);
    // init the draw buffer
    {
        int size_per_vertices = 8;
        int size_per_triangle = 3 * size_per_vertices;
        mTriangleDrawBuffer.resize(size_per_triangle * mTriangleArray.size());

        int size_per_edge = 2 * size_per_vertices;
        mEdgesDrawBuffer.resize(size_per_edge * mEdgeArray.size());
    }

    CalcTriangleDrawBuffer();
    CalcEdgesDrawBuffer();

    // init the inv mass vector
    mInvMassMatrixDiag.noalias() = tVectorXd::Zero(GetNumOfFreedom());
    for (int i = 0; i < mVertexArray.size(); i++)
    {
        mInvMassMatrixDiag.segment(i * 3, 3).fill(1.0 / mVertexArray[i]->mMass);
    }

    mVcur.noalias() = tVectorXd::Zero(GetNumOfFreedom());

    // SIM_INFO("init geo done");
    // exit(0);
}

extern void CalcTriangleDrawBufferSingle(tVertex *v0, tVertex *v1, tVertex *v2,
                                         tVectorXf &buffer, int &st_pos);
extern void CalcEdgeDrawBufferSingle(tVertex *v0, tVertex *v1, tVectorXf &buffer,
                                     int &st_pos);

void cTrimeshScene::CalcTriangleDrawBuffer()
{
    mTriangleDrawBuffer.fill(std::nan(""));
    int st = 0;
    for (auto &x : mTriangleArray)
    {
        CalcTriangleDrawBufferSingle(
            mVertexArray[x->mId0],
            mVertexArray[x->mId1],
            mVertexArray[x->mId2],
            mTriangleDrawBuffer,
            st);
    }
}

void cTrimeshScene::CalcEdgesDrawBuffer()
{
    mEdgesDrawBuffer.fill(std::nan(""));
    int st = 0;
    for (auto &x : mEdgeArray)
    {
        CalcEdgeDrawBufferSingle(
            mVertexArray[x->mId0],
            mVertexArray[x->mId1],
            mEdgesDrawBuffer, st);
    }
}

/**
 * \brief           Update substeps
*/
void cTrimeshScene::UpdateSubstep()
{
    // std::cout << "[before update] x = " << mXcur.transpose() << std::endl;
    // exit(0);
    // std::cout << "update sub step " << mCurdt << std::endl;
    switch (mScheme)
    {
    case eIntegrationScheme::TRI_POSITION_BASED_DYNAMIC:
        UpdateSubstepPBD();
        break;
    case eIntegrationScheme::TRI_BARAFF:
        SIM_ERROR("baraff hasn't been impled");
        break;
    case eIntegrationScheme::TRI_PROJECTIVE_DYNAMIC:
        UpdateSubstepProjDyn();
        break;
    default:
        SIM_ERROR("unsupported scheme {}", mScheme);
        break;
    }
    // std::cout << "xcur res = " << mXcur.transpose() << std::endl;
}

/**
 * \brief           Update for position based dynamics
*/
void cTrimeshScene::UpdateSubstepPBD()
{
    ClearForce();

    // 1. calc ext force
    CalcExtForce(mExtForce);
    // 2. update unconstrained
    UpdateVelAndPosUnconstrained(mExtForce);
    /*
        3. collision detect
            build constraint
    */
    // ConstraintSetupPBD();

    // 4. solve constraint
    ConstraintProcessPBD();

    // 5. post process vel
    PostProcessPBD();
}

/**
 * \brief           Update the unconstrained vel and pos 
*/
void cTrimeshScene::UpdateVelAndPosUnconstrained(const tVectorXd &fext)
{
    // std::cout << "fext = " << fext.transpose() << std::endl;
    mVcur += mInvMassMatrixDiag.cwiseProduct(fext) * mCurdt;
    // std::cout << "mVcur = " << mVcur.transpose() << std::endl;
    mXcur += mVcur * mCurdt;
    // std::cout << "mXcur = " << mXcur.transpose() << std::endl;
}
// /**
//  * \brief           create the constraint for PBD
// */
// void cTrimeshScene::ConstraintSetupPBD()
// {

// }

/**
 * \brief           given raw vertex vector p, solve the constraint and get the new p
*/

void cTrimeshScene::ConstraintProcessPBD()
{
    const int iters = mItersPBD;
    double raw_k = mStiffnessPBD;
    double final_k = 1 - std::pow((1 - raw_k), 1.0 / iters);
    const bool enable_strech_constraint = true; // for each edge
    // std::cout << "X cur = " << mXcur.transpose() << std::endl;
    // std::cout << "mEdgeArray size = " << mEdgeArray.size() << std::endl;
    for (int i = 0; i < iters; i++)
    {
        if (mEnableParallelPBD == true)
        {
#pragma omp parallel for
            for (int e_id = 0; e_id < mEdgeArray.size(); e_id++)
            {
                auto e = mEdgeArray[e_id];
                if (enable_strech_constraint)
                {
                    int id0 = e->mId0, id1 = e->mId1;
                    const tVector3d &p1 = mXcur.segment(3 * id0, 3),
                                    &p2 = mXcur.segment(3 * id1, 3);
                    double raw = e->mRawLength;
                    // std::cout << "raw = " << raw << std::endl;
                    double dist = (p1 - p2).norm();
                    double w1 = mInvMassMatrixDiag[3 * e->mId0],
                           w2 = mInvMassMatrixDiag[3 * e->mId1];
                    double w_sum = w1 + w2;
                    double coef1 = -w1 / w_sum * final_k,
                           coef2 = w2 / w_sum * final_k;
                    if (w_sum == 0)
                    {
                        continue;
                    }
                    tVector3d delta_p1 = coef1 * (dist - raw) * (p1 - p2) / dist,
                              delta_p2 = coef2 * (dist - raw) * (p1 - p2) / dist;
                    // #pragma omp ordered
                    // std::cout << "vertex " << id0 << " += " << delta_p1.segment(0, 3).transpose() << std::endl;
                    // #pragma omp critical
                    {
                        mXcur[3 * id0 + 0] += delta_p1[0];
                        mXcur[3 * id0 + 1] += delta_p1[1];
                        mXcur[3 * id0 + 2] += delta_p1[2];

                        mXcur[3 * id1 + 0] += delta_p2[0];
                        mXcur[3 * id1 + 1] += delta_p2[1];
                        mXcur[3 * id1 + 2] += delta_p2[2];
                    }
                }
            }
        }
        else
        {
            for (int e_id = 0; e_id < mEdgeArray.size(); e_id++)
            {
                auto e = mEdgeArray[e_id];
                if (enable_strech_constraint)
                {
                    int id0 = e->mId0, id1 = e->mId1;
                    const tVector3d &p1 = mXcur.segment(3 * id0, 3),
                                    &p2 = mXcur.segment(3 * id1, 3);
                    double raw = e->mRawLength;
                    // std::cout << "raw = " << raw << std::endl;
                    double dist = (p1 - p2).norm();
                    double w1 = mInvMassMatrixDiag[3 * e->mId0],
                           w2 = mInvMassMatrixDiag[3 * e->mId1];
                    double w_sum = w1 + w2;
                    double coef1 = -w1 / w_sum * final_k,
                           coef2 = w2 / w_sum * final_k;
                    if (w_sum == 0)
                    {
                        continue;
                    }
                    tVector3d delta_p1 = coef1 * (dist - raw) * (p1 - p2) / dist,
                              delta_p2 = coef2 * (dist - raw) * (p1 - p2) / dist;
                    // #pragma omp ordered
                    // std::cout << "vertex " << id0 << " += " << delta_p1.segment(0, 3).transpose() << std::endl;
                    // #pragma omp critical
                    {
                        mXcur[3 * id0 + 0] += delta_p1[0];
                        mXcur[3 * id0 + 1] += delta_p1[1];
                        mXcur[3 * id0 + 2] += delta_p1[2];

                        mXcur[3 * id1 + 0] += delta_p2[0];
                        mXcur[3 * id1 + 1] += delta_p2[1];
                        mXcur[3 * id1 + 2] += delta_p2[2];
                    }
                }
            }
        }
    }
}

/**
 * \brief           
*/
void cTrimeshScene::PostProcessPBD()
{
    // SIM_WARN("PostProcessPBD hasn't been impled");
    mVcur = (mXcur - mXpre) / mCurdt;
    UpdateCurNodalPosition(mXcur);
    mXpre = mXcur;
}

void cTrimeshScene::InitConstraint(const Json::Value &root)
{
    cSimScene::InitConstraint(root);

    for (auto &i : mFixedPointIds)
    {
        // std::cout << "fixed id = " << i << std::endl;
        mInvMassMatrixDiag.segment(i * 3, 3).setZero();
    }
}

#include "utils/JsonUtil.h"

void cTrimeshScene::Init(const std::string &conf_path)
{
    Json::Value root;
    cJsonUtil::LoadJson(conf_path, root);
    mGeometryType = cJsonUtil::ParseAsString("geometry_type", root);
    cSimScene::Init(conf_path);

    if (mScheme == eIntegrationScheme::TRI_POSITION_BASED_DYNAMIC)
    {
        Json::Value pbd_config = cJsonUtil::ParseAsValue("pbd_config", root);
        mItersPBD = cJsonUtil::ParseAsInt("max_pbd_iters", pbd_config);
        mStiffnessPBD = cJsonUtil::ParseAsDouble("stiffness_pbd", pbd_config);
        mEnableParallelPBD = cJsonUtil::ParseAsBool("enable_parallel_pbd", pbd_config);
    }
    InitGeometry(root);
    InitConstraint(root);

    // 3. set up the init pos
    CalcNodePositionVector(mXpre);
    mXcur.noalias() = mXpre;
}

/**
 * \brief               Update substep for projective dynamic
*/
void cTrimeshScene::UpdateSubstepProjDyn()
{
}

bool SetColor(int edge_id, int num_of_colors, const int max_edge_id,
              int *edge_color_info, std::vector<std::set<int>> &color_vertices, const tEigenArr<tEdge *> &edge_info_array)
{
    if (edge_id >= max_edge_id)
    {
        std::cout << "division done for " << edge_id << " edges\n";
        return true;
        // exit(0);
    }
    else
    {
        // confirm the edge has no color
        SIM_ASSERT(edge_color_info[edge_id] == -1);
        // get the vertices id of this edge
        int v0 = edge_info_array[edge_id]->mId0,
            v1 = edge_info_array[edge_id]->mId1;
        // begin to divide this edge
        for (int i = 0; i < num_of_colors; i++)
        {
            auto &res = color_vertices[i];
            bool valid = (res.find(v0) == res.end()) && (res.find(v1) == res.end());
            if (valid)
            {
                // set the color, push its vertices into the set
                edge_color_info[edge_id] = i;
                res.insert(v0);
                res.insert(v1);

                // continue to the next edge, check whehter it's succ
                bool succ = SetColor(edge_id + 1, num_of_colors, max_edge_id, edge_color_info, color_vertices, edge_info_array);
                if (succ)
                    // if succ, break
                    return true;
                else
                {
                    res.erase(v0);
                    res.erase(v1);
                    edge_color_info[edge_id] = -1;
                    // else if failed, remove my vertices from the set, continue
                }
            }
            else
            {
                // if not valid, then the color is not suitable, continues
                continue;
            }
        }
    }
    // if all colors is impossible, return
    return false;
}

void cTrimeshScene::CalcExtForce(tVectorXd &ext_force) const
{
    cSimScene::CalcExtForce(ext_force);
    ext_force += -mDamping * this->mVcur;
    // std::cout << "damping = " << (-mDamping * this->mVcur).transpose() << std::endl;
}