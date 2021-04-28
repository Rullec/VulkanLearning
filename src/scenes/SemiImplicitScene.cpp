#include "SemiImplicitScene.h"
#include "geometries/Primitives.h"
#include "utils/JsonUtil.h"
#include "utils/TimeUtil.hpp"
#include <set>
#include <iostream>
cSemiImplicitScene::cSemiImplicitScene() { mEdgeArray.clear(); }

cSemiImplicitScene::~cSemiImplicitScene()
{
    for (auto x : mEdgeArray)
        delete x;
    mEdgeArray.clear();
}

void cSemiImplicitScene::Init(const std::string &conf_path)
{
    Json::Value root;
    cJsonUtil::LoadJson(conf_path, root);

    cSimScene::Init(conf_path);

    mEnableQBending = cJsonUtil::ParseAsBool("enable_Q_bending", root);
    mBendingStiffness = cJsonUtil::ParseAsDouble("bending_stiffness", root);
    // 2. create geometry, dot allocation
    InitGeometry(root);
    InitRaycaster();
    InitConstraint(root);
    InitDrawBuffer();

    if (mEnableQBending)
        InitBendingHessian();
    // 3. set up the init pos
    CalcNodePositionVector(mXpre);
    mXcur.noalias() = mXpre;
    SIM_INFO("Init simulation scene done");
}

// Dense deprecated
// /**
//  * \brief           calculate the matrix used in fast simulation
//  *      x_a^i - x_bi = Si * x
//  *      J = [k1S1T; k2S2T, ... knSnT]
//  *      Jinv = J.inverse()
//  *      L = \sum_i ki * Si^T * Si
//  *      (M + dt2 * L).inv()
// */
// void cSemiImplicitScene::InitVarsOptImplicit()
// {
//     int num_of_sprs = GetNumOfEdges();
//     int node_dof = GetNumOfFreedom();
//     int spr_dof = 3 * num_of_sprs;
//     J.noalias() = tMatrixXd::Zero(node_dof, spr_dof);
//     tMatrixXd L = tMatrixXd::Zero(node_dof, node_dof);

//     tMatrixXd Si = tMatrixXd::Zero(3, node_dof);
//     for (int i = 0; i < num_of_sprs; i++)
//     {
//         // 1. calc Si
//         auto spr = mSpringArray[i];
//         int id0 = spr->mId0, id1 = spr->mId1;
//         double k = spr->mK_spring;
//         Si.setZero();
//         Si.block(0, 3 * id0, 3, 3).setIdentity();
//         Si.block(0, 3 * id1, 3, 3).noalias() = tMatrix3d::Identity(3, 3) * -1;

//         J.block(0, 3 * i, node_dof, 3).noalias() = k * Si.transpose();
//         L += k * Si.transpose() * Si;
//     }
//     // std::cout << "L=\n"
//     //           << L << std::endl;
//     // std::cout << "Minv * L=\n"
//     //           << mInvMassMatrixDiag.asDiagonal().toDenseMatrix() * L << std::endl;
//     double dt2 = mIdealDefaultTimestep * mIdealDefaultTimestep;
//     SIM_ASSERT(dt2 > 0);
//     I_plus_dt2_Minv_L_inv = (tMatrixXd::Identity(node_dof, node_dof) + dt2 * mInvMassMatrixDiag.asDiagonal().toDenseMatrix() * L).inverse();
//     // std::cout << "J = \n"
//     //           << J << std::endl;
//     // std::cout << "Jinv = \n"
//     //           << Jinv << std::endl;
//     // std::cout << "(I + dt2 Minv L).inv = \n"
//     //           << I_plus_dt2_Minv_L_inv << std::endl;
//     SIM_INFO("init vars succ");
//     // exit(0);
// }

void cSemiImplicitScene::Update(double dt) { cSimScene::Update(dt); }

void cSemiImplicitScene::Reset() { cSimScene::Reset(); }

void cSemiImplicitScene::UpdateSubstep()
{
    // std::cout << "-----------------\n";
    if (mEnableProfiling == true)
        cTimeUtil::Begin("substep");
    SIM_ASSERT(std::fabs(mCurdt - mIdealDefaultTimestep) < 1e-10);
    // std::cout << "[update] x = " << mXcur.transpose() << std::endl;
    // 1. clear force
    ClearForce();

    // std::cout << "mInt force = " << mIntForce.transpose() << std::endl;
    // std::cout << "mExt force = " << mExtForce.transpose() << std::endl;
    // 3. forward simulation
    tVectorXd mXnext = tVectorXd::Zero(GetNumOfFreedom());

    // 2. calculate force
    CalcIntForce(mXcur, mIntForce);
    CalcExtForce(mExtForce);
    CalcDampingForce((mXcur - mXpre) / mCurdt, mDampingForce);

    // std::cout << "before x = " << mXcur.transpose() << std::endl;
    // std::cout << "fint = " << mIntForce.transpose() << std::endl;
    // std::cout << "fext = " << mExtForce.transpose() << std::endl;
    // std::cout << "fdamp = " << mDampingForce.transpose() << std::endl;
    mXnext = CalcNextPositionSemiImplicit();

    // std::cout << "mXnext = " << mXnext.transpose() << std::endl;
    mXpre.noalias() = mXcur;
    mXcur.noalias() = mXnext;
    UpdateCurNodalPosition(mXcur);
    if (mEnableProfiling == true)
        cTimeUtil::End("substep");
}

/**
 * \brief       Given total force, calcualte the next vertices' position
*/
tVectorXd cSemiImplicitScene::CalcNextPositionSemiImplicit() const
{
    /*
        semi implicit
        X_next = dt2 * Minv * Ftotal + 2 * Xcur - Xpre
    */

    double dt2 = mCurdt * mCurdt;
    tVectorXd next_pos = dt2 * mInvMassMatrixDiag.cwiseProduct(
                                   mIntForce + mExtForce + mDampingForce) +
                         2 * mXcur - mXpre;

    return next_pos;
}

void cSemiImplicitScene::InitConstraint(const Json::Value &root)
{
    cSimScene::InitConstraint(root);

    // mass modification
    for (auto &i : mFixedPointIds)
    {
        mInvMassMatrixDiag.segment(i * 3, 3).setZero();
        // printf("[debug] fixed point id %d at ", i);
        // exit(0);
        // next_pos.segment(i * 3, 3) = mXcur.segment(i * 3, 3);
        // std::cout << mXcur.segment(i * 3, 3).transpose() << std::endl;
    }
    // std::cout << "inv mass = " << mInvMassMatrixDiag.transpose() << std::endl;
    // exit(0);
}

#include "omp.h"
#include <iostream>
/**
 * \brief   discretazation from square cloth to mass spring system
 * 
*/
#include "geometries/Triangulator.h"
#include "utils/JsonUtil.h"
void cSemiImplicitScene::InitGeometry(const Json::Value &conf)
{
    cSimScene::InitGeometry(conf);
    // int gap = mSubdivision + 1;

    // set up the vertex pos data
    // in XOY plane

    mStiffness = cJsonUtil::ParseAsDouble("stiffness", conf);
    for (auto &x : mEdgeArray)
        x->mK_spring = mStiffness;
}

/**
 * \brief           calcualte the Q bending internal force
*/
void cSemiImplicitScene::CalcIntForce(const tVectorXd &xcur, tVectorXd &int_force) const
{
    if (mEnableProfiling == true)
        cTimeUtil::Begin("fint");
    if (mEnableProfiling == true)
        cTimeUtil::Begin("fint_sim");
    cSimScene::CalcIntForce(xcur, int_force);
    if (mEnableProfiling == true)
        cTimeUtil::End("fint_sim");
    if (mEnableProfiling == true)
        cTimeUtil::Begin("fint_bend");
    if (mEnableQBending == true)
    {
        // std::cout << "[debug] begin to calculate the bending force in inextensible assumption\n";
        const tVectorXd &f_bending = -mBendingHessianQ * xcur;
        int_force += f_bending;
        // std::cout << "bend force = " << f_bending.transpose() << std::endl;
    }
    if (mEnableProfiling == true)
        cTimeUtil::End("fint_bend");
    if (mEnableProfiling == true)
        cTimeUtil::End("fint");
}

/**
 * \brief           Calculate the bending hessian (constant in inextensible assumption)
 * 
 * 
 *          E_{bending} = 0.5 xT * Q x
 * 
 * Q \in [node_dof, node_dof]
 * 
 * According to the discrete mean curvature normal operator, for a edge "ei" with two adjoint triangles, the bending energy hessian matrix can be defined as:
 * 
 * "ei" connect two points: x0 and x1.
 *      triangle1: x0 x1 x2
 *      triangle2: x0 x1 x3
 * 
 * Five edges in these two triangles
 *          e0 = x0 - x1
 *          e1 = x0 - x2
 *          e2 = x0 - x3
 *          e3 = x1 - x2
 *          e4 = x1 - x3
 * Four angles between adjoint edges
 *          t01 = theta<e0, e1>
 *          t02 = theta<e0, e2>
 *          t03 = theta<e0, e3>
 *          t04 = theta<e0, e4>
 *          
 * cotangent values:
 *          c01 = cot(t01)
 *          c02 = cot(t02)
 *          c03 = cot(t03)
 *          c04 = cot(t04)
 * 
 * Ki = [(c03​+c04​)I3, ​​(c01​+c02​)I3​​, −(c01​+c03​)I3​​, −(c02​+c04​)I3​​]  \in R^{3 \times 12}
 *    = [Ki0, Ki1, Ki2, Ki3]
 * E = 0.5 x^T KiT * Ki x
 * Qi = KiT * Ki \in R^{12 \times 12}
 *    = [
 *  Ki0 * K.T 
 *  Ki1 * K.T 
 *  Ki2 * K.T 
 *  Ki3 * K.T 
 * ]  is assigned to
 * [
 *  (x0_id, x0_id) & (x0_id, x1_id) & (x0_id, x2_id) & (x0_id, x3_id) \\
 *  (x1_id, x0_id) & (x1_id, x1_id) & (x1_id, x2_id) & (x1_id, x3_id) \\
 *  (x2_id, x0_id) & (x2_id, x1_id) & (x2_id, x2_id) & (x2_id, x3_id) \\
 *  (x3_id, x0_id) & (x3_id, x1_id) & (x3_id, x2_id) & (x3_id, x3_id) \\
 * ]
 * 
 * It's a symmetric matrix
 * 
*/
int SelectAnotherVerteix(tTriangle *tri, int v0, int v1)
{
    SIM_ASSERT(tri != nullptr);
    std::set<int> vid_set = {
        tri->mId0,
        tri->mId1,
        tri->mId2};
    // printf("[debug] select another vertex in triangle 3 vertices (%d, %d, %d) besides %d %d\n", tri->mId0, tri->mId1, tri->mId2, v0, v1);
    vid_set.erase(vid_set.find(v0));
    vid_set.erase(vid_set.find(v1));
    return *vid_set.begin();
};

tVector CalculateCotangentCoeff(const tVector &x0,
                                tVector &x1,
                                tVector &x2,
                                tVector &x3)
{
    const tVector &e0 = x0 - x1,
                  &e1 = x0 - x2,
                  &e2 = x0 - x3,
                  &e3 = x1 - x2,
                  &e4 = x1 - x3;
    // std::cout << "e0 = " << e0.transpose() << std::endl;
    // std::cout << "e1 = " << e1.transpose() << std::endl;
    // std::cout << "e2 = " << e2.transpose() << std::endl;
    // std::cout << "e3 = " << e3.transpose() << std::endl;
    // std::cout << "e4 = " << e4.transpose() << std::endl;
    const double &e0_norm = e0.norm(),
                 e1_norm = e1.norm(),
                 e2_norm = e2.norm(),
                 e3_norm = e3.norm(),
                 e4_norm = e4.norm();
    // printf("[debug] norm: e0 = %.3f, e1 = %.3f, e2 = %.3f, e3 = %.3f, e4 = %.3f\n",
    //        e0_norm,
    //        e1_norm,
    //        e2_norm,
    //        e3_norm,
    //        e4_norm);
    const double &t01 = std::acos(std::fabs(e0.dot(e1)) / (e0_norm * e1_norm)),
                 &t02 = std::acos(std::fabs(e0.dot(e2)) / (e0_norm * e2_norm)),
                 &t03 = std::acos(std::fabs(e0.dot(e3)) / (e0_norm * e3_norm)),
                 &t04 = std::acos(std::fabs(e0.dot(e4)) / (e0_norm * e4_norm));
    // const double &t01 = std::acos(e0.dot(e1) / (e0_norm * e1_norm)),
    //              &t02 = std::acos(e0.dot(e2) / (e0_norm * e2_norm)),
    //              &t03 = std::acos(e0.dot(e3) / (e0_norm * e3_norm)),
    //              &t04 = std::acos(e0.dot(e4) / (e0_norm * e4_norm));
    // printf("[debug] theta: t01 = %.3f, t02 = %.3f, t03 = %.3f, t04 = %.3f\n",
    //        t01, t02, t03, t04);
    const double &c01 = 1.0 / std::tan(t01),
                 &c02 = 1.0 / std::tan(t02),
                 &c03 = 1.0 / std::tan(t03),
                 &c04 = 1.0 / std::tan(t04);
    return tVector(c03 + c04, c01 + c02, -c01 - c03, -c02 - c04);
}
void cSemiImplicitScene::InitBendingHessian()
{
    // std::cout << "---------\n";
    int dof = GetNumOfFreedom();
    mBendingHessianQ.resize(dof, dof);

    std::vector<tTriplet> Q_trilet_array(0);
    for (int i = 0; i < mEdgeArray.size(); i++)
    {
        const auto &e = this->mEdgeArray[i];
        if (e->mIsBoundary == false)
        {
            // printf("[debug] bending, tri %d and tri %d, shared edge: %d\n",
            //        e->mTriangleId0, e->mTriangleId1, i);
            int vid[4] = {e->mId0,
                          e->mId1,
                          SelectAnotherVerteix(mTriangleArray[e->mTriangleId0], e->mId0, e->mId1),
                          SelectAnotherVerteix(mTriangleArray[e->mTriangleId1], e->mId0, e->mId1)};
            // printf("[debug] bending, tri %d and tri %d, shared edge: %d, total vertices: %d %d %d %d\n",
            //        e->mTriangleId0, e->mTriangleId1, i, vid[0], vid[1], vid[2], vid[3]);
            tVector cot_vec = CalculateCotangentCoeff(
                mVertexArray[vid[0]]->mPos,
                mVertexArray[vid[1]]->mPos,
                mVertexArray[vid[2]]->mPos,
                mVertexArray[vid[3]]->mPos);
            // std::cout << "cot vec = " << cot_vec.transpose() << std::endl;
            for (int row_id = 0; row_id < 4; row_id++)
                for (int col_id = 0; col_id < 4; col_id++)
                {
                    double square = 1.0 / mTriangleArray.size() * 2;
                    double value = mBendingStiffness * cot_vec[row_id] * cot_vec[col_id] * 3 / square;
                    for (int j = 0; j < 3; j++)
                    {
                        Q_trilet_array.push_back(tTriplet(3 * vid[row_id] + j, 3 * vid[col_id] + j, value));
                    }
                }
        }
    }
    mBendingHessianQ.setFromTriplets(Q_trilet_array.begin(), Q_trilet_array.end());

    // std::cout << "bending Q constant done = \n"
    //           << mBendingHessianQ << std::endl;
    printf("bending Q constant done %d/%d, bending stiffnes = %.3f\n", Q_trilet_array.size(), mBendingHessianQ.size(), mBendingStiffness);
    // exit(0);
}