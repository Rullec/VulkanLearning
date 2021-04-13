#include "PDScene.h"
#include "geometries/Primitives.h"
#include "utils/JsonUtil.h"
#include <iostream>

cPDScene::cPDScene()
{
    mMaxSteps_Opt = 0;
}

void cPDScene::Init(const std::string &conf_path)
{

    Json::Value root;
    cJsonUtil::LoadJson(conf_path, root);

    cSimScene::Init(conf_path);
    mMaxSteps_Opt = cJsonUtil::ParseAsInt("max_steps_opt", root);
    InitGeometry(root);
    InitConstraint(root);

    // 3. set up the init pos
    CalcNodePositionVector(mXpre);
    mXcur.noalias() = mXpre;
    InitVarsOptImplicitSparse();
    I_plus_dt2_Minv_L_sparse_solver.analyzePattern(I_plus_dt2_Minv_L_sparse);
    I_plus_dt2_Minv_L_sparse_solver.factorize(I_plus_dt2_Minv_L_sparse);
}

void cPDScene::InitGeometry(const Json::Value &conf)
{
    cSimScene::InitGeometry(conf);
    double mStiffness = cJsonUtil::ParseAsDouble("stiffness", conf);
    for (auto &e : mEdgeArray)
    {
        e->mK_spring = mStiffness;
    }
}
/**
 * \brief           initialize sparse variables for paper "fast simulation"
*/
void cPDScene::InitVarsOptImplicitSparse()
{
    int num_of_sprs = GetNumOfEdges();
    int node_dof = GetNumOfFreedom();
    int spr_dof = 3 * num_of_sprs;
    // 1. initialize J (set up the triplet array)
    J_sparse.resize(node_dof, spr_dof);
    I_plus_dt2_Minv_L_sparse.resize(node_dof, node_dof);
    tEigenArr<tTriplet> J_sparse_tri_lst(0);
    tEigenArr<tTriplet> I_plus_dt2_Minv_L(0);
    for (int i = 0; i < GetNumOfFreedom(); i++)
    {
        I_plus_dt2_Minv_L.push_back(tTriplet(i, i, 1));
    }
    for (int i = 0; i < num_of_sprs; i++)
    {
        auto spr = mEdgeArray[i];
        int id0 = spr->mId0;
        int id1 = spr->mId1;

        // J sparse
        {
            for (int j = 0; j < 3; j++)
                J_sparse_tri_lst.push_back(
                    tTriplet(3 * id0 + j, i * 3 + j, 1 * spr->mK_spring));
            for (int j = 0; j < 3; j++)
                J_sparse_tri_lst.push_back(
                    tTriplet(3 * id1 + j, i * 3 + j, -1 * spr->mK_spring));
        }
        /*
            dt2 * Minv * L
            L = [   cola:   colb:
          rowa:      I    &  -I
          rowb:      -I   &   I
            ]

            dt2 * Minv * L = 
            k2 * 
            [
                           cola:                   colb:
          rowa:      dt2 * Minva *k2I    &  -dt2 * Minva * k2I
          rowb:      -dt2 * Minvb * k2I  &    dt2 * Minvb * k2I
            ]
        */
        double minva = mInvMassMatrixDiag[3 * id0],
               minvb = mInvMassMatrixDiag[3 * id1];
        double dt2 = mIdealDefaultTimestep * mIdealDefaultTimestep;
        // std::cout << "ideal dt = " << mIdealDefaultTimestep << std::endl;
        // exit(0);
        // double dt2 = 1;
        double k = spr->mK_spring;
        for (int j = 0; j < 3; j++)
        {
            {
                // 0 0
                I_plus_dt2_Minv_L.push_back(
                    tTriplet(3 * id0 + j, 3 * id0 + j, dt2 * minva * k * 1));
                // 1 1
                I_plus_dt2_Minv_L.push_back(
                    tTriplet(3 * id1 + j, 3 * id1 + j, dt2 * minvb * k * 1));
                // 0 1
                I_plus_dt2_Minv_L.push_back(
                    tTriplet(3 * id0 + j, 3 * id1 + j, -dt2 * minva * k * 1));
                // 1 0
                I_plus_dt2_Minv_L.push_back(
                    tTriplet(3 * id1 + j, 3 * id0 + j, -dt2 * minvb * k * 1));
            }
        }
    }
    J_sparse.setFromTriplets(J_sparse_tri_lst.begin(), J_sparse_tri_lst.end());
    // 2. init I_plus_dt2_Minv_Linv
    I_plus_dt2_Minv_L_sparse.setFromTriplets(I_plus_dt2_Minv_L.begin(),
                                             I_plus_dt2_Minv_L.end());
}

/**
 * \brief           calculat next position by optimization implciit method (fast simulation)
 * 
 *      1. set up the init solution, caluclate the b
 *      2. begin to do iteration
 *      3. return the result
*/
#include "utils/TimeUtil.hpp"
tVectorXd cPDScene::CalcNextPositionOptImplicit() const
{

    // cTimeUtil::Begin("fast simulation calc next");
    // std::cout << "begin CalcNextPositionOptImplicit\n";
    tVectorXd y = 2 * mXcur - mXpre;
    tVectorXd Xnext = y;
    tVectorXd d = tVectorXd::Zero(3 * GetNumOfEdges());

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
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
        for (int j = 0; j < GetNumOfEdges(); j++)
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

void cPDScene::UpdateSubstep()
{
    ClearForce();

    const tVectorXd &Xnext = CalcNextPositionOptImplicit();
    mXpre.noalias() = mXcur;
    mXcur.noalias() = Xnext;
    UpdateCurNodalPosition(mXcur);
}

void cPDScene::Reset() { cSimScene::Reset(); }

void cPDScene::Update(double dt) { cSimScene::Update(dt); }