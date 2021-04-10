#include "SemiImplicitScene.h"
#include "geometries/Primitives.h"
#include "utils/JsonUtil.h"
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
    // 2. create geometry, dot allocation


    InitGeometry(root);
    InitConstraint(root);

    // 3. set up the init pos
    CalcNodePositionVector(mXpre);
    mXcur.noalias() = mXpre;
    std::cout << "init sim scene done\n";
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
    // std::cout << "-----------------\n";
    // std::cout << "before x = " << mXcur.transpose() << std::endl;
    // std::cout << "fint = " << mIntForce.transpose() << std::endl;
    // std::cout << "fext = " << mExtForce.transpose() << std::endl;
    // std::cout << "fdamp = " << mDampingForce.transpose() << std::endl;
    mXnext = CalcNextPositionSemiImplicit();

    // std::cout << "mXnext = " << mXnext.transpose() << std::endl;
    mXpre.noalias() = mXcur;
    mXcur.noalias() = mXnext;
    UpdateCurNodalPosition(mXcur);
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

// -----------implicit methods-------------
void cSemiImplicitScene::PushState(const std::string &name) const {}

void cSemiImplicitScene::PopState(const std::string &name) {}

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