#include "TrimeshScene.h"
#include <atomic>
#include <omp.h>
#include "utils/LogUtil.h"
#include <set>
#include <iostream>

tTriangle::tTriangle()
{
    mId0 = mId1 = mId2 = -1;
}
tTriangle::tTriangle(int a, int b, int c) : mId0(a), mId1(b), mId2(c)

{
}

tEdge::tEdge()
{
    mId0 = mId1 = -1;
    mRawLength = -1;
    mIsBoundary = false;
    mTriangleId0 = mTriangleId1 = -1;
}

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
void cTrimeshScene::InitGeometry()
{
    int num_of_lines = (mSubdivision + 1); // = 3
    int num_of_vertices = num_of_lines * num_of_lines;
    int num_of_edges = num_of_lines * mSubdivision * 2 + mSubdivision * mSubdivision;
    int num_of_triangles = num_of_lines * num_of_lines * 2;

    mVertexArray.clear();
    mEdgeArray.clear();
    mTriangleArray.clear();

    // 1. init the triangles
    for (int row_id = 0; row_id < mSubdivision; row_id++)
    {
        for (int col_id = 0; col_id < mSubdivision; col_id++)
        {
            int up_left = row_id * num_of_lines + col_id;
            auto tri1 = new tTriangle(up_left, up_left + num_of_lines, up_left + 1 + num_of_lines);
            auto tri2 = new tTriangle(up_left, up_left + 1 + num_of_lines, up_left + 1);

            mTriangleArray.push_back(tri1);
            // printf("[debug] triangle %d vertices %d %d %d\n", mTriangleArray.size() - 1, tri1->mId0, tri1->mId1, tri1->mId2);
            mTriangleArray.push_back(tri2);
            // printf("[debug] triangle %d vertices %d %d %d\n", mTriangleArray.size() - 1, tri2->mId0, tri2->mId1, tri2->mId2);
        }
    }
    printf("[debug] create %d triangles done\n", mTriangleArray.size());
    // 2. init the vertices
    double unit_edge_length = mClothWidth / mSubdivision;
    double unit_mass = mClothMass / (num_of_lines * num_of_lines);
    {
        for (int i = 0; i < num_of_lines; i++)
            for (int j = 0; j < num_of_lines; j++)
            {
                tVertex *v = new tVertex();
                v->mMass = unit_mass;
                v->mPos =
                    tVector(unit_edge_length * j, mClothWidth - unit_edge_length * i, 0, 1) + mClothInitPos;
                v->mPos[3] = 1;
                v->mColor = tVector(0, 196.0 / 255, 1, 0);
                mVertexArray.push_back(v);
                v->muv =
                    tVector2f(i * 1.0 / mSubdivision, j * 1.0 / mSubdivision);
                // printf("create vertex %d at (%.7f, %.7f), uv (%.7f, %.7f)\n",
                //        mVertexArray.size() - 1, v->mPos[0], v->mPos[1],
                //        v->muv[0], v->muv[1]);
            }
    }
    printf("[debug] create %d vertices done\n", mVertexArray.size());
    // 3. init the edges
    {
        int num_of_edges_per_line = mSubdivision * 3 + 1;
        int num_of_vertices_per_line = num_of_lines;
        int num_of_triangles_per_line = 2 * mSubdivision;
        for (int row_id = 0; row_id < mSubdivision + 1; row_id++)
        {
            // 3.1 add top line
            for (int col_id = 0; col_id < mSubdivision; col_id++)
            {
                // edge id: num_of_edges_per_line * row_id + col_id
                tEdge *edge = new tEdge();
                edge->mId0 = num_of_vertices_per_line * row_id + col_id;
                edge->mId1 = edge->mId0 + 1;
                edge->mRawLength = unit_edge_length;
                if (row_id == 0)
                {
                    edge->mIsBoundary = true;
                    edge->mTriangleId0 = 2 * col_id + 1;
                }
                else if (row_id == mSubdivision)
                {
                    edge->mIsBoundary = true;
                    edge->mTriangleId0 = num_of_triangles_per_line * (mSubdivision - 1) + col_id * 2;
                }
                else
                {
                    edge->mIsBoundary = false;
                    edge->mTriangleId0 = num_of_triangles_per_line * (row_id - 1) + 2 * col_id;
                    edge->mTriangleId1 = num_of_triangles_per_line * row_id + 2 * col_id + 1;
                }
                mEdgeArray.push_back(edge);
                // printf("[debug] edge %d v0 %d v1 %d, is boundary %d, triangle0 %d, triangle1 %d\n",
                //    mEdgeArray.size() - 1, edge->mId0, edge->mId1, edge->mIsBoundary, edge->mTriangleId0, edge->mTriangleId1);
            }
            if (row_id == mSubdivision)
                break;
            // 3.2 add middle lines

            for (int col_counting_id = 0; col_counting_id < 2 * mSubdivision + 1; col_counting_id++)
            {
                int col_id = col_counting_id / 2;
                tEdge *edge = new tEdge();
                if (col_counting_id % 2 == 0)
                {
                    // vertical line
                    edge->mId0 = row_id * num_of_vertices_per_line + col_id;
                    edge->mId1 = (row_id + 1) * num_of_vertices_per_line + col_id;
                    edge->mRawLength = unit_edge_length;
                    if (col_id == 0)
                    {
                        // left edge
                        edge->mIsBoundary = true;
                        edge->mTriangleId0 = num_of_triangles_per_line * row_id;
                    }
                    else if (col_counting_id == 2 * mSubdivision)
                    {
                        // right edge
                        edge->mIsBoundary = true;
                        edge->mTriangleId0 = num_of_triangles_per_line * (row_id + 1) - 1;
                    }
                    else
                    {
                        // middle edges
                        edge->mIsBoundary = false;
                        edge->mTriangleId0 = num_of_triangles_per_line * row_id + col_id;
                        edge->mTriangleId1 = num_of_triangles_per_line * row_id + col_id + 1;
                    }
                }
                else
                {
                    continue;
                    std::cout << "ignore skew edge\n";
                    // skew line
                    edge->mId0 = num_of_vertices_per_line * row_id + col_id;
                    edge->mId1 = num_of_vertices_per_line * (row_id + 1) + col_id + 1;
                    edge->mIsBoundary = false;
                    edge->mRawLength = unit_edge_length * std::sqrt(2);
                    edge->mTriangleId0 = num_of_triangles * row_id + col_id * 2;
                    edge->mTriangleId1 = edge->mTriangleId0 + 1;
                }
                mEdgeArray.push_back(edge);
                // printf("[debug] edge %d v0 %d v1 %d, is boundary %d, triangle0 %d, triangle1 %d\n",
                //    mEdgeArray.size() - 1, edge->mId0, edge->mId1, edge->mIsBoundary, edge->mTriangleId0, edge->mTriangleId1);
            }
        }
    }
    printf("[debug] create %d edges done\n", mEdgeArray.size());
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
#pragma omp parallel for num_thread(12)
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
    mItersPBD = cJsonUtil::ParseAsInt("max_pbd_iters", root);
    mStiffnessPBD = cJsonUtil::ParseAsDouble("stiffness_pbd", root);
    mEnableParallelPBD = cJsonUtil::ParseAsBool("enable_parallel_pbd", root);
    cSimScene::Init(conf_path);
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