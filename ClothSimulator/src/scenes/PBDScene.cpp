#include "PBDScene.h"
#include "geometries/Primitives.h"
#include "utils/LogUtil.h"
#include <atomic>
#include <iostream>
#include <omp.h>
#include <set>
extern int SelectAnotherVerteix(tTriangle *tri, int v0, int v1);

extern tVector CalculateCotangentCoeff(const tVector &x0,
                                       tVector &x1,
                                       tVector &x2,
                                       tVector &x3);

cPBDScene::cPBDScene()
{
    mRayArray.clear();
    mTriangleArray.clear();
    mEdgeArray.clear();
    mVcur.resize(0);
    mBendingMatrixKArray.clear();
}

/**
 * \brief           Calculat ethe bending constant matrix "K" for each edge
 * 
 * if one edge is located in the boundary, its value will be "Nan"
*/
void cPBDScene::InitBendingMatrixPBD()
{
    mBendingMatrixKArray.resize(mEdgeArray.size(), tVector::Ones() * std::nan(""));
    for (int i = 0; i < mEdgeArray.size(); i++)
    {
        const auto &e = mEdgeArray[i];
        if (e->mIsBoundary == false)
        {
            // it's an interior boundary, begin to calculate the K
            int vid[4] = {e->mId0,
                          e->mId1,
                          SelectAnotherVerteix(mTriangleArray[e->mTriangleId0], e->mId0, e->mId1),
                          SelectAnotherVerteix(mTriangleArray[e->mTriangleId1], e->mId0, e->mId1)};
            // printf("[debug] bending, tri %d and tri %d, shared edge: %d, total vertices: %d %d %d %d\n",
            //        e->mTriangleId0, e->mTriangleId1, i, vid[0], vid[1], vid[2], vid[3]);
            mBendingMatrixKArray[i].noalias() = CalculateCotangentCoeff(
                mVertexArray[vid[0]]->mPos,
                mVertexArray[vid[1]]->mPos,
                mVertexArray[vid[2]]->mPos,
                mVertexArray[vid[3]]->mPos);
            // std::cout << "[debug] pbd K for edge " << i << " = " << mBendingMatrixKArray[i].transpose() << std::endl;
        }
    }
    // exit(0);
}
cPBDScene::~cPBDScene()
{
    for (auto &x : mEdgeArray)
        delete x;
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
void cPBDScene::InitGeometry(const Json::Value &conf)
{
    cTriangulator::BuildGeometry(conf, mVertexArray, mEdgeArray,
                                 mTriangleArray);
    // init the draw buffer
    {
        int size_per_vertices = 8;
        int size_per_triangle = 3 * size_per_vertices;
        mTriangleDrawBuffer.resize(size_per_triangle * mTriangleArray.size());

        int size_per_edge = 2 * size_per_vertices;
        // mEdgesDrawBuffer.resize(size_per_edge * mEdgeArray.size());
        mEdgesDrawBuffer.resize(size_per_edge *
                                (mEdgeArray.size() + mMaxDrawRayDebug));
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

    if (mEnableBendingPBD == true)
    {
        InitBendingMatrixPBD();
    }
    // SIM_INFO("init geo done");
    // exit(0);
}

extern void CalcTriangleDrawBufferSingle(tVertex *v0, tVertex *v1, tVertex *v2,
                                         tVectorXf &buffer, int &st_pos);
extern void CalcEdgeDrawBufferSingle(tVertex *v0, tVertex *v1,
                                     tVectorXf &buffer, int &st_pos);
extern void CalcEdgeDrawBufferSingle(const tVector &v0, const tVector &v1,
                                     tVectorXf &buffer, int &st_pos);

void cPBDScene::CalcTriangleDrawBuffer()
{
    mTriangleDrawBuffer.fill(std::nan(""));
    int st = 0;
    for (auto &x : mTriangleArray)
    {
        CalcTriangleDrawBufferSingle(
            mVertexArray[x->mId0], mVertexArray[x->mId1], mVertexArray[x->mId2],
            mTriangleDrawBuffer, st);
    }
}

void cPBDScene::CalcEdgesDrawBuffer()
{
    mEdgesDrawBuffer.fill(std::nan(""));
    int st = 0;
    for (auto &x : mEdgeArray)
    {
        CalcEdgeDrawBufferSingle(mVertexArray[x->mId0], mVertexArray[x->mId1],
                                 mEdgesDrawBuffer, st);
    }
    // std::cout << "calc edges draw buffer size = " << mRayArray.size()
    //           << std::endl;
    for (auto &x : mRayArray)
    {
        CalcEdgeDrawBufferSingle(x->mOrigin, x->mOrigin + x->mDir * 10,
                                 mEdgesDrawBuffer, st);
    }
}

/**
 * \brief           Update substeps
*/
void cPBDScene::UpdateSubstep()
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
void cPBDScene::UpdateSubstepPBD()
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
void cPBDScene::UpdateVelAndPosUnconstrained(const tVectorXd &fext)
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
// void cPBDScene::ConstraintSetupPBD()
// {

// }

/**
 * \brief           solve the stretch (spring) constraint for PBD
 * \param final_k   the determined constraint stiffness
*/
void cPBDScene::StretchConstraintProcessPBD(double final_k)
{

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (int e_id = 0; e_id < mEdgeArray.size(); e_id++)
    {
        auto e = mEdgeArray[e_id];
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
        const tVector3d &delta_p1 =
                            coef1 * (dist - raw) * (p1 - p2) / dist,
                        &delta_p2 =
                            coef2 * (dist - raw) * (p1 - p2) / dist;
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
/**
 * \brief           given raw vertex vector p, solve the constraint and get the new p
*/
void cPBDScene::ConstraintProcessPBD()
{
    const int iters = mItersPBD;
    double stretch_k = 1 - std::pow((1 - mStiffnessPBD), 1.0 / iters);
    double bending_k = 1 - std::pow((1 - mBendingStiffnessPBD), 1.0 / iters);
    // std::cout << "X cur = " << mXcur.transpose() << std::endl;
    // std::cout << "mEdgeArray size = " << mEdgeArray.size() << std::endl;
    for (int i = 0; i < iters; i++)
    {
        StretchConstraintProcessPBD(stretch_k);

        if (mEnableBendingPBD == true)
        {
            BendingConstraintProcessPBD(bending_k);
        }
        // std::cout << "X cur = " << mXcur.transpose() << std::endl;
        // exit(0);
    }
}

/**
 * \brief           process the bendign constraint 
 * \param final_k   determined PBD stiffness
*/
void cPBDScene::BendingConstraintProcessPBD(double final_k)
{
    tMatrixXd K = tMatrixXd::Zero(3, 12);
    // for (auto &e : mEdgeArray)
    for (int _i = 0; _i < mEdgeArray.size(); _i++)
    {
        const auto &e = mEdgeArray[_i];
        if (e->mIsBoundary == false)
        {
            // for interior edges, calculate the nodal displacement "delta"
            /*
                for a given interior edge (i.e. a pair of adjoint triangles),
                the bending constraint C(x) = 0 is\in R^3 
                C(x) = [C_0, C_1, C_2] = [C_i], i = {0, 1, 2}
                Here we solve C_i(x) \in R individually.
            */
            int vid[4] = {e->mId0,
                          e->mId1,
                          SelectAnotherVerteix(mTriangleArray[e->mTriangleId0], e->mId0, e->mId1),
                          SelectAnotherVerteix(mTriangleArray[e->mTriangleId1], e->mId0, e->mId1)};
            tVectorXd x_total = tVectorXd::Zero(12);

            const tVector &K_vec = mBendingMatrixKArray[_i];
            for (int j = 0; j < 4; j++)
            {
                K.block(0, 3 * j, 3, 3).noalias() = tMatrix3d::Identity() * K_vec[j];
                x_total.segment(3 * j, 3).noalias() = mXcur.segment(vid[j] * 3, 3);
            }
            // std::cout << "K = \n"
            //           << K << std::endl;
            // std::cout << "x_total = " << x_total.transpose() << std::endl;
            tVector inv_mass_vec = tVector(
                mInvMassMatrixDiag[3 * vid[0]],
                mInvMassMatrixDiag[3 * vid[1]],
                mInvMassMatrixDiag[3 * vid[2]],
                mInvMassMatrixDiag[3 * vid[3]]);
            // std::cout << "inv mass vec = " << inv_mass_vec.transpose() << std::endl;
            for (int j = 0; j < 3; j++) // j is the constraint id in [0,2]
            {
                /*
                begin to Solve C_j
               */
                // SIM_ERROR("hasn't been impled");
                // 1. calculate C_j(x)
                double C_j = K.row(j).dot(x_total);
                // printf("[debug] C_%d = %.3f\n", j, C_j);
                // 1. calculate \nabla_{x_k} C_j(x), k=[0, 1, 2, 3]
                tVector3d DCj_Dxk[4] = {K.row(j).segment(3 * 0, 3),
                                        K.row(j).segment(3 * 1, 3),
                                        K.row(j).segment(3 * 2, 3),
                                        K.row(j).segment(3 * 3, 3)};

                // 2. calculate s_j
                double s_j = 0;
                for (int k = 0; k < 4; k++)
                {
                    s_j += (inv_mass_vec[k] * DCj_Dxk[k].norm());
                }
                s_j = C_j / s_j;
                // printf("[debug] s%d = %.3f\n", j, s_j);
                // 3. calculate \Delta x_i^j
                for (int i = 0; i < 4; i++)
                {
                    int id = vid[i];
                    tVector3d delta = -s_j * inv_mass_vec[i] * DCj_Dxk[i] * final_k;
                    // std::cout << "DC" << j << "Dx" << i << " = " << delta.transpose() << std::endl;
                    mXcur.segment(id * 3, 3) += delta;
                }
            }
        }
    }
}
/**
 * \brief           
*/
void cPBDScene::PostProcessPBD()
{
    // SIM_WARN("PostProcessPBD hasn't been impled");
    mVcur = (mXcur - mXpre) / mCurdt;
    UpdateCurNodalPosition(mXcur);
    mXpre = mXcur;
}

void cPBDScene::InitConstraint(const Json::Value &root)
{
    cSimScene::InitConstraint(root);

    for (auto &i : mFixedPointIds)
    {
        // std::cout << "fixed id = " << i << std::endl;
        mInvMassMatrixDiag.segment(i * 3, 3).setZero();
    }
}

#include "utils/JsonUtil.h"

void cPBDScene::Init(const std::string &conf_path)
{
    Json::Value root;
    cJsonUtil::LoadJson(conf_path, root);
    cSimScene::Init(conf_path);

    Json::Value pbd_config = cJsonUtil::ParseAsValue("pbd_config", root);
    mItersPBD = cJsonUtil::ParseAsInt("max_pbd_iters", pbd_config);
    mStiffnessPBD = cJsonUtil::ParseAsDouble("stiffness_pbd", pbd_config);
    mEnableParallelPBD =
        cJsonUtil::ParseAsBool("enable_parallel_pbd", pbd_config);
    mEnableBendingPBD = cJsonUtil::ParseAsBool("enable_bending_pbd", pbd_config);
    mBendingStiffnessPBD = cJsonUtil::ParseAsDouble("bending_stiffness_pbd", pbd_config);
    printf("[pbd] enable bending %d, bendign stiffness %.4f\n", mEnableBendingPBD, mBendingStiffnessPBD);

    InitGeometry(root);
    InitConstraint(root);

    // 3. set up the init pos
    CalcNodePositionVector(mXpre);
    mXcur.noalias() = mXpre;
}

/**
 * \brief               Update substep for projective dynamic
*/
void cPBDScene::UpdateSubstepProjDyn() {}

bool SetColor(int edge_id, int num_of_colors, const int max_edge_id,
              int *edge_color_info, std::vector<std::set<int>> &color_vertices,
              const tEigenArr<tEdge *> &edge_info_array)
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
            bool valid =
                (res.find(v0) == res.end()) && (res.find(v1) == res.end());
            if (valid)
            {
                // set the color, push its vertices into the set
                edge_color_info[edge_id] = i;
                res.insert(v0);
                res.insert(v1);

                // continue to the next edge, check whehter it's succ
                bool succ =
                    SetColor(edge_id + 1, num_of_colors, max_edge_id,
                             edge_color_info, color_vertices, edge_info_array);
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

void cPBDScene::CalcExtForce(tVectorXd &ext_force) const
{
    cSimScene::CalcExtForce(ext_force);
    ext_force += -mDamping * this->mVcur;
    // std::cout << "damping = " << (-mDamping * this->mVcur).transpose() << std::endl;
}
