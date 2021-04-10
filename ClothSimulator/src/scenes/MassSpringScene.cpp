#include "MassSpringScene.h"
#include "geometries/Primitives.h"
#include "utils/JsonUtil.h"
#include <iostream>
cMSScene::cMSScene() { mEdgeArray.clear(); }

cMSScene::~cMSScene()
{
    for (auto x : mEdgeArray)
        delete x;
    mEdgeArray.clear();
}

void cMSScene::Init(const std::string &conf_path)
{
    Json::Value root;
    cJsonUtil::LoadJson(conf_path, root);

    cSimScene::Init(conf_path);
    // 2. create geometry, dot allocation

    if (mScheme == eIntegrationScheme::MS_IMPLICIT)
    {
        mMaxNewtonIters = cJsonUtil::ParseAsInt("max_newton_iters", root);
    }
    if (mScheme == eIntegrationScheme::MS_OPT_IMPLICIT)
    {
        mMaxSteps_Opt = cJsonUtil::ParseAsInt("max_steps_opt", root);
    }

    InitGeometry(root);
    InitConstraint(root);

    // 3. set up the init pos
    CalcNodePositionVector(mXpre);
    mXcur.noalias() = mXpre;

    if (mScheme == eIntegrationScheme::MS_OPT_IMPLICIT)
    {
        // InitVarsOptImplicit();
        InitVarsOptImplicitSparse();
        I_plus_dt2_Minv_L_sparse_solver.analyzePattern(
            I_plus_dt2_Minv_L_sparse);
        I_plus_dt2_Minv_L_sparse_solver.factorize(I_plus_dt2_Minv_L_sparse);
        // std::cout << "factorize done\n";
        // exit(0);
        // tMatrixXd res = (I_plus_dt2_Minv_L_inv * I_plus_dt2_Minv_L_sparse - tMatrixXd::Identity(GetNumOfFreedom(), GetNumOfFreedom()));

        // std::cout << "J = \n"
        //           << J << std::endl;
        // std::cout << "J sparse = \n"
        //           << J_sparse << std::endl;
        // std::cout << "diff = " << (J - J_sparse).norm() << std::endl;

        // std::cout << "res norm = \n"
        //           << res.norm() << std::endl;
        // exit(0);
    }
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
// void cMSScene::InitVarsOptImplicit()
// {
//     int num_of_sprs = GetNumOfSprings();
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

void cMSScene::Update(double dt) { cSimScene::Update(dt); }

void cMSScene::Reset() { cSimScene::Reset(); }

void cMSScene::UpdateSubstep()
{
    SIM_ASSERT(std::fabs(mCurdt - mIdealDefaultTimestep) < 1e-10);
    // std::cout << "[update] x = " << mXcur.transpose() << std::endl;
    // 1. clear force
    ClearForce();

    // std::cout << "mInt force = " << mIntForce.transpose() << std::endl;
    // std::cout << "mExt force = " << mExtForce.transpose() << std::endl;
    // 3. forward simulation
    tVectorXd mXnext = tVectorXd::Zero(GetNumOfFreedom());
    switch (mScheme)
    {
    case eIntegrationScheme::MS_SEMI_IMPLICIT:
    {
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
        // std::cout << "after x = " << mXnext.transpose() << std::endl;
        // exit(0);
        break;
    }
    case eIntegrationScheme::MS_IMPLICIT:
        mXnext = CalcNextPositionImplicit();
        break;
    case eIntegrationScheme::MS_OPT_IMPLICIT:
        mXnext = CalcNextPositionOptImplicit();
        break;
    default:
        SIM_ERROR("Unsupported integration scheme {}", mScheme);
        break;
    }
    // std::cout << "mXnext = " << mXnext.transpose() << std::endl;
    mXpre.noalias() = mXcur;
    mXcur.noalias() = mXnext;
    UpdateCurNodalPosition(mXcur);
}

/**
 * \brief       Given total force, calcualte the next vertices' position
*/
tVectorXd cMSScene::CalcNextPositionSemiImplicit() const
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

/**
 * \brief           Calcualte Xnext by implicit integration, which means, solve the system equation by newton iteration
*/
#include <Eigen/SparseLU>
tVectorXd cMSScene::CalcNextPositionImplicit()
{
    /*
        Let X = Xnext is what we want have, the equation is:

        G(X) = dt2 * F_int(X) - M * X + M (2 * Xcur - Xpre) + dt2 * Fext = 0

        Let x0 = Xcur as an init solution (we can certainly have some other init x0)
        Let i = 0
        1. Calc res = G(Xi), if i > max_iters or |res| < thre, to step5; else step 2
        2. Xi <- Xi - (dGdX).inv() * res, i+=1, to step1
        5. return res
    */

    int max_iters = mMaxNewtonIters;
    double cnvg_thre = 1e-4; // convergence threshold
    tVectorXd x0 = mXcur;
    int dof = GetNumOfFreedom();
    tVectorXd res = tVectorXd::Zero(dof);
    tSparseMat dGdx(dof, dof);
    int cur_iter = 0;

    Eigen::SparseLU<tSparseMat> solver;

    while (true)
    {
        // step 1
        CalcGxImplicit(x0, res, mIntForce, mExtForce, mDampingForce);
        double res_norm = res.norm();
        if (cur_iter > max_iters || res_norm < cnvg_thre)
            break;

        // step 2
        // CalcdGxdxImplicit(x0, dGdx);
        CalcdGxdxImplicitSparse(x0, dGdx);
        // TestdGxdxImplicit(x0, dGdx);

        // if (cur_iter == 3)
        // {
        //     tSparseMat mat(dof, dof);
        //     CalcdGxdxImplicitSparse(x0, mat);
        //     std::cout << "diff norm = " << (mat - dGdx).norm() << std::endl;
        //     std::cout << "dense = \n"
        //               << dGdx << std::endl;
        //     std::cout << "sparse = \n"
        //               << mat << std::endl;
        //     exit(0);
        // }
        if (cur_iter == 0)
            solver.analyzePattern(dGdx);

        solver.factorize(dGdx);
        //Use the factors to solve the linear system
        // x = ;
        x0 = x0 - solver.solve(res);
        cur_iter++;
        // printf("iter %d diff %.5f\n", cur_iter, res_norm);
        if (x0.hasNaN())
        {
            std::cout << "x0 has Nan = " << x0.transpose() << std::endl;
            std::cout << "x0 init = " << mXcur.transpose() << std::endl;
            // std::cout << "dGdx = \n"
            //           << dGdx << std::endl;
            // std::cout << "dGdx inv = \n"
            //           << dGdx.inverse() << std::endl;

            exit(0);
        }
        // std::cout << "iter " << cur_iter << " res = " << res.norm()
        //           << std::endl;
        // exit(0);
    }

    printf("[debug] newton solver done, iters %d/%d, res norm %.5f\n", cur_iter,
           max_iters, res.norm());
    // exit(0);
    return x0;
}

void cMSScene::InitConstraint(const Json::Value &root)
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
/**
 * \brief           given current pos, calculate the value of dynamic equation
 *      G(x) = 2 * Xcur - Xpre - X + Minv * dt2 * (Fext + Fint) = 0
*/
void cMSScene::CalcGxImplicit(const tVectorXd &x, tVectorXd &Gx,
                              tVectorXd &fint_buffer, tVectorXd &fext_buffer,
                              tVectorXd &damp_buffer) const
{
    CalcIntForce(x, fint_buffer);
    CalcExtForce(fext_buffer);
    CalcDampingForce((mXcur - mXpre) / mCurdt, damp_buffer);
    Gx.noalias() = 2 * mXcur - mXpre - x +
                   mCurdt * mCurdt *
                       mInvMassMatrixDiag.cwiseProduct(
                           fext_buffer + fint_buffer + damp_buffer);
}

/**
 * \brief               Given x, Calculate d(Gx)/dx 
*/
void cMSScene::CalcdGxdxImplicit(const tVectorXd &x, tMatrixXd &dGdx) const
{
    dGdx.noalias() = tMatrixXd::Zero(GetNumOfFreedom(), GetNumOfFreedom());
    int id0, id1;
    double dist;
    tVector3d pos0, pos1;
    for (auto &spr : this->mEdgeArray)
    {
        id0 = spr->mId0;
        id1 = spr->mId1;

        /*
            For a "l"th spring, the node index of its two ends is "i" and "j"
            let \bar{f}_i is the force contribution to "i"th node
            let \bar{f}_j is the force contribution to "j"th node
            l_{raw} is the raw length of this spring

            Calc:
                d(\bar{f}_i) / dxi, d(\bar{f}_i) / dxj
        
            dfi/dxi = k * I3 
                    - k * l_{raw} *
                        (
                            I_3 * |xi - xj| - (xi - xj) * (xi - xj)^T / |xi - xj|
                        )
                        /
                        (xi - xj)^2
        
            dfi/dxj = - dfi/dxi
            dfj/dxi = - dfi/dxi
            dfj/dxj = - dfi/dxj = dfidxi

            In summary:
            dFdX should be a real symmetric matrix
        */
        pos0 = x.segment(3 * id0, 3);
        pos1 = x.segment(3 * id1, 3);
        double dist = (pos0 - pos1).norm();
        const tMatrix3d &I3 = tMatrix3d::Identity(3, 3);
        tMatrix3d dfidxi =
            spr->mK_spring * I3 -
            spr->mK_spring * spr->mRawLength *
                (I3 * dist - (pos0 - pos1) * (pos0 - pos1).transpose() / dist) /
                (dist * dist);

        if (dfidxi.hasNaN() == true)
        {
            printf("df%ddx%d has Nan, id0 = %d, id1 = %d, dist = %.5f\n", id0,
                   id0, id0, id1, dist);
            std::cout << "dfidxi = \n" << dfidxi << std::endl;
            std::cout << "pos0 = " << pos0.transpose() << std::endl;
            std::cout << "pos1 = " << pos1.transpose() << std::endl;
            std::cout << "x = " << x.transpose() << std::endl;
            exit(0);
        }
        dGdx.block(3 * id0, 3 * id0, 3, 3) += dfidxi;
        dGdx.block(3 * id1, 3 * id1, 3, 3) += dfidxi;
        dGdx.block(3 * id0, 3 * id1, 3, 3) += -dfidxi;
        dGdx.block(3 * id1, 3 * id0, 3, 3) += -dfidxi;
    }

    // std::cout << "dFdx = \n" << dGdx << std::endl;
    // dGdx = dt2 * Minv * dFdX - I
    dGdx = mCurdt * mCurdt * mInvMassMatrixDiag.asDiagonal().toDenseMatrix() *
               dGdx -
           tMatrixXd::Identity(GetNumOfFreedom(), GetNumOfFreedom());
}

/**
 * \brief           Given x0, test whehter the analytic gradient is correct
*/
void cMSScene::TestdGxdxImplicit(const tVectorXd &x0, const tMatrixXd &Gx_ana)
{
    double eps = 1e-6;
    tVectorXd Gx_old, Gx_new;
    tVectorXd x_now = x0;
    CalcGxImplicit(x_now, Gx_old, mIntForce, mExtForce, mDampingForce);

    for (int i = 0; i < x0.size(); i++)
    {
        x_now[i] += eps;
        CalcGxImplicit(x_now, Gx_new, mIntForce, mExtForce, mDampingForce);
        tVectorXd dGdxi_num = (Gx_new - Gx_old) / eps;
        tVectorXd dGdxi_ana = Gx_ana.col(i);
        tVectorXd diff = dGdxi_ana - dGdxi_num;
        double diff_norm = diff.norm();
        if (diff_norm > 10 * eps)
        {
            printf("[error] error test dGdx/dx%d failed, diff norm %.5f\n", i,
                   diff_norm);
            exit(0);
        }

        x_now[i] -= eps;
    }
    SIM_INFO("test dG/dx succ");
}

void cMSScene::CalcdGxdxImplicitSparse(const tVectorXd &x,
                                       tSparseMat &dGdx) const
{
    int dof = GetNumOfFreedom();
    Eigen::SparseMatrix<double> spMat(dof, dof);
    int id0, id1;
    double dist;
    tVector3d pos0, pos1;
    double dt2 = mCurdt * mCurdt;
    tEigenArr<tTriplet> tri_lst(0);
    for (auto &spr : this->mEdgeArray)
    {
        id0 = spr->mId0;
        id1 = spr->mId1;

        pos0 = x.segment(3 * id0, 3);
        pos1 = x.segment(3 * id1, 3);
        double dist = (pos0 - pos1).norm();
        const tMatrix3d &I3 = tMatrix3d::Identity(3, 3);
        tMatrix3d dfidxi =
            spr->mK_spring * I3 -
            spr->mK_spring * spr->mRawLength *
                (I3 * dist - (pos0 - pos1) * (pos0 - pos1).transpose() / dist) /
                (dist * dist);

        if (dfidxi.hasNaN() == true)
        {
            printf("df%ddx%d has Nan, id0 = %d, id1 = %d, dist = %.5f\n", id0,
                   id0, id0, id1, dist);
            std::cout << "dfidxi = \n" << dfidxi << std::endl;
            std::cout << "pos0 = " << pos0.transpose() << std::endl;
            std::cout << "pos1 = " << pos1.transpose() << std::endl;
            std::cout << "x = " << x.transpose() << std::endl;
            exit(0);
        }
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
            {
                double val = dfidxi(i, j);
                {
                    tri_lst.push_back(tTriplet(3 * id0 + i, 3 * id0 + j, val));
                    tri_lst.push_back(tTriplet(3 * id1 + i, 3 * id1 + j, val));
                    tri_lst.push_back(tTriplet(3 * id0 + i, 3 * id1 + j, -val));
                    tri_lst.push_back(tTriplet(3 * id1 + i, 3 * id0 + j, -val));
                }
            }
    }
    dGdx.setFromTriplets(tri_lst.begin(), tri_lst.end());
    dGdx = dt2 * mInvMassMatrixDiag.asDiagonal() * dGdx;

    for (int i = 0; i < GetNumOfFreedom(); i++)
    {
        dGdx.coeffRef(i, i) -= 1;
    }
    //     tri_lst.push_back(tTriplet(i, i, -1));
    // dGdx.resize(GetNumOfFreedom(), GetNumOfFreedom());
    // dGdx.setFromTriplets(tri_lst.begin(), tri_lst.end());
    /*
        dt2 * inv
    */
    // std::cout << "dFdx = \n" << dGdx << std::endl;
    // dGdx = dt2 * Minv * dFdX - I
    // dGdx = mCurdt * mCurdt * mInvMassMatrixDiag.asDiagonal().toDenseMatrix() *
    //            dGdx -
    //        tMatrixXd::Identity(GetNumOfFreedom(), GetNumOfFreedom());
}
void cMSScene::PushState(const std::string &name) const {}

void cMSScene::PopState(const std::string &name) {}

/**
 * \brief           add damping forces
*/
void cMSScene::CalcDampingForce(const tVectorXd &vel, tVectorXd &damping) const
{
    damping.noalias() = -vel * mDamping;
}

/**
 * \brief           calculat next position by optimization implciit method (fast simulation)
 * 
 *      1. set up the init solution, caluclate the b
 *      2. begin to do iteration
 *      3. return the result
*/
#include "utils/TimeUtil.hpp"
tVectorXd cMSScene::CalcNextPositionOptImplicit() const
{

    // cTimeUtil::Begin("fast simulation calc next");
    // std::cout << "begin CalcNextPositionOptImplicit\n";
    tVectorXd y = 2 * mXcur - mXpre;
    tVectorXd Xnext = y;
    tVectorXd d = tVectorXd::Zero(3 * GetNumOfSprings());

    // 1. calculate b = dt2 * fext - M * y
    // y = 2 * xcur - xpre
    tVectorXd fext = tVectorXd::Zero(GetNumOfFreedom());
    tVectorXd fdamping = tVectorXd::Zero(GetNumOfFreedom());
    CalcExtForce(fext);

    CalcDampingForce((mXcur - mXpre) / mCurdt, fdamping);
    fext += fdamping;
    // tVectorXd b;
    double dt2 = mCurdt * mCurdt;

    SIM_ASSERT(std::fabs(mCurdt - mIdealDefaultTimestep) < 1e-10);
    // std::cout << "max step = " << mMaxSteps_Opt << std::endl;
    for (int i = 0; i < mMaxSteps_Opt; i++)
    {
        /*
            1. fixed x, calculate the d
            d = [d1, d2, ... dn]
            di = (x0^i - x1^i).normalized()
        */
        // std::cout << "step " << i << " X = " << Xnext.transpose() << std::endl;
#pragma omp parallel for
        for (int j = 0; j < GetNumOfSprings(); j++)
        {
            int id0 = mEdgeArray[j]->mId0, id1 = mEdgeArray[j]->mId1;
            d.segment(j * 3, 3).noalias() =
                (Xnext.segment(3 * id0, 3) - Xnext.segment(3 * id1, 3))
                    .normalized() *
                mEdgeArray[j]->mRawLength;
        }
        // std::cout << "d = " << d.transpose() << std::endl;
        // std::cout << "J * d = " << (J * d).transpose() << std::endl;
        // std::cout << "b= " << b.transpose() << std::endl;
        /*
            2. fixed the d, calulcate the x
            x = (M + dt2 * L).inv() * (dt2 * J * d - b)
        */
        // cTimeUtil::BeginLazy("fast simulation sparse solve");
        Xnext.noalias() = I_plus_dt2_Minv_L_sparse_solver.solve(
            mInvMassMatrixDiag.cwiseProduct(dt2 * (J_sparse * d + fext)) + y);
        // cTimeUtil::EndLazy("fast simulation sparse solve");
        //  = I_plus_dt2_Minv_L_inv * ();
        if (Xnext.hasNaN())
        {
            std::cout << "Xnext has Nan, exit = " << Xnext.transpose()
                      << std::endl;
            exit(0);
        }
    }
    // cTimeUtil::ClearLazy("fast simulation sparse solve");
    // std::cout << "done, xnext = " << Xnext.transpose() << std::endl;
    // exit(0);
    // cTimeUtil::End("fast simulation calc next");
    return Xnext;
}

int cMSScene::GetNumOfSprings() const { return mEdgeArray.size(); }