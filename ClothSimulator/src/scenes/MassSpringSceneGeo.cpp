#include "MassSpringScene.h"
#include "omp.h"
#include <iostream>
#include "geometries/Primitives.h"
/**
 * \brief   discretazation from square cloth to mass spring system
 * 
*/
#include "geometries/Triangulator.h"
#include "utils/JsonUtil.h"
void cMSScene::InitGeometry(const Json::Value &conf)
{

    // int gap = mSubdivision + 1;

    // set up the vertex pos data
    // in XOY plane
    std::vector<tTriangle *> tmp;
    cTriangulator::BuildGeometry(conf, mVertexArray, mEdgeArray, tmp);

    mStiffness = cJsonUtil::ParseAsDouble("stiffness", conf);
    for (auto &x : mEdgeArray)
        x->mK_spring = mStiffness;
    // init the buffer
    {
        int num_of_triangles = tmp.size();
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

/**
 * \brief       internal force
*/
#include <atomic>
void cMSScene::CalcIntForce(const tVectorXd &xcur, tVectorXd &int_force) const
{
    // std::vector<std::atomic<double>> int_force_atomic(int_force.size());
    // for (int i = 0; i < int_force.size(); i++)
    //     int_force_atomic[i] = 0;
    // double res = 1;
    // std::vector<double> int_force_atomic(int_force.size());

    // #pragma omp parallel for
    // std::cout << "input fint = " << int_force.transpose() << std::endl;
    int id0, id1;
    double dist;
#pragma omp parallel for private(id0, id1, dist) num_threads(12)
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
        // #pragma omp critical
        //         std::cout << "spring " << i << " force = " << force0.transpose() << ", dist " << dist << ", v0 " << id0 << " v1 " << id1 << std::endl;
        // std::cout << "spring " << i << ", v0 = " << id0 << " v1 = " << id1 << std::endl;
        // 2. add force
        {
#pragma omp atomic
            int_force[3 * id0 + 0] += force0[0];
            int_force[3 * id0 + 1] += force0[1];
            int_force[3 * id0 + 2] += force0[2];
// #pragma omp atomic
// #pragma omp atomic
#pragma omp atomic
            int_force[3 * id1 + 0] += -force0[0];
            int_force[3 * id1 + 1] += -force0[1];
            int_force[3 * id1 + 2] += -force0[2];
            // #pragma omp atomic
            // #pragma omp atomic
            // #pragma omp atomic
            //             inf_force_0 += force0;
            // #pragma omp atomic
            //             inf_force_1 += force1;
        }
    }
    // std::cout << "output fint = " << int_force.transpose() << std::endl;
    // exit(0);
}

/**
 * \brief           initialize sparse variables for paper "fast simulation"
*/
void cMSScene::InitVarsOptImplicitSparse()
{
    int num_of_sprs = GetNumOfSprings();
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
                J_sparse_tri_lst.push_back(tTriplet(3 * id0 + j, i * 3 + j, 1 * spr->mK_spring));
            for (int j = 0; j < 3; j++)
                J_sparse_tri_lst.push_back(tTriplet(3 * id1 + j, i * 3 + j, -1 * spr->mK_spring));
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
    J_sparse.setFromTriplets(
        J_sparse_tri_lst.begin(),
        J_sparse_tri_lst.end());
    // 2. init I_plus_dt2_Minv_Linv
    I_plus_dt2_Minv_L_sparse.setFromTriplets(I_plus_dt2_Minv_L.begin(),
                                             I_plus_dt2_Minv_L.end());
}