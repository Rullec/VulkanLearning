#include "PDScene.h"
#include "geometries/Primitives.h"
#include "utils/JsonUtil.h"
#include "utils/TimeUtil.hpp"
#include <iostream>

extern int SelectAnotherVerteix(tTriangle *tri, int v0, int v1);
extern tVector CalculateCotangentCoeff(const tVector &x0, tVector &x1,
                                       tVector &x2, tVector &x3);

cPDScene::cPDScene() { mMaxSteps_Opt = 0; }

void cPDScene::Init(const std::string &conf_path)
{

    Json::Value root;
    cJsonUtil::LoadJson(conf_path, root);

    cSimScene::Init(conf_path);
    mMaxSteps_Opt = cJsonUtil::ParseAsInt("max_steps_opt", root);
    mEnableBending = cJsonUtil::ParseAsBool("enable_bending", root);
    mBendingStiffness = cJsonUtil::ParseAsDouble("bending_stiffness", root);
    InitGeometry(root);
    InitRaycaster();
    InitConstraint(root);
    InitDrawBuffer();

    // 3. set up the init pos
    CalcNodePositionVector(mXpre);
    mXcur.noalias() = mXpre;
    InitVarsOptImplicitSparse();
    InitVarsOptImplicitSparseFast();
    I_plus_dt2_Minv_L_sparse_solver.analyzePattern(I_plus_dt2_Minv_L_sparse);
    I_plus_dt2_Minv_L_sparse_solver.factorize(I_plus_dt2_Minv_L_sparse);
    I_plus_dt2_Minv_L_sparse_fast_solver.analyzePattern(
        I_plus_dt2_Minv_L_sparse_fast);
    I_plus_dt2_Minv_L_sparse_fast_solver.factorize(
        I_plus_dt2_Minv_L_sparse_fast);
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
    // 1. Identity
    for (int i = 0; i < GetNumOfFreedom(); i++)
    {
        I_plus_dt2_Minv_L.push_back(tTriplet(i, i, 1));
    }

    // 2. J sparse and 2h2
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

    // 3. get bending system matrix contribution
    if (mEnableBending)
    {
        std::cout << "[debug] add bending!\n";
        AddBendTriplet(I_plus_dt2_Minv_L);
        // I_plus_dt2_Minv_L.insert(I_plus_dt2_Minv_L.begin(),
        //                          bending_triplet.begin(),
        //                          bending_triplet.end());
    }
    // 4. init the matrices by triplets
    J_sparse.setFromTriplets(J_sparse_tri_lst.begin(), J_sparse_tri_lst.end());
    // 2. init I_plus_dt2_Minv_Linv
    I_plus_dt2_Minv_L_sparse.setFromTriplets(I_plus_dt2_Minv_L.begin(),
                                             I_plus_dt2_Minv_L.end());
}

void cPDScene::InitVarsOptImplicitSparseFast()
{
    int dof_3 = GetNumOfVertices();
    I_plus_dt2_Minv_L_sparse_fast.resize(dof_3, dof_3);
    std::vector<tTriplet> tri_lst(0);
    for (int k = 0; k < I_plus_dt2_Minv_L_sparse.outerSize(); k++)
    {
        for (tSparseMat::InnerIterator it(I_plus_dt2_Minv_L_sparse, k); it;
             ++it)
        {
            // printf("row %d col %d value %.4f\n", it.row(),
            //        it.col(), it.value());
            if ((it.row() % 3 == 0) && (it.col() % 3 == 0))
            {
                tri_lst.push_back(
                    tTriplet(it.row() / 3, it.col() / 3, it.value()));
            }
        }
    }
    I_plus_dt2_Minv_L_sparse_fast.setFromTriplets(tri_lst.begin(),
                                                  tri_lst.end());
    // std::cout << "old = \n " << I_plus_dt2_Minv_L_sparse << std::endl;
    // std::cout << "new = \n " << I_plus_dt2_Minv_L_sparse_fast << std::endl;

    // exit(0);
}
/**
 * \brief           calculat next position by optimization implciit method (fast
 * simulation)
 *
 *      1. set up the init solution, caluclate the b
 *      2. begin to do iteration
 *      3. return the result
 */
template <typename T>
void SolveFast(const tSparseMat &A, const T &solver, tVectorXd &residual,
               tVectorXd &solution)
{
    int size = residual.size();
    if (solution.size() != size)
        solution.resize(size);
    Eigen::Map<tVectorXd, 0, Eigen::InnerStride<3>> res0(residual.data(),
                                                         size / 3);
    Eigen::Map<tVectorXd, 0, Eigen::InnerStride<3>> res1(residual.data() + 1,
                                                         size / 3);
    Eigen::Map<tVectorXd, 0, Eigen::InnerStride<3>> res2(residual.data() + 2,
                                                         size / 3);

    // Eigen::Map<tVectorXd, 0, Eigen::InnerStride<3>> sol0(solution.data(),
    // size / 3); Eigen::Map<tVectorXd, 0, Eigen::InnerStride<3>>
    // sol1(solution.data() + 1, size / 3); Eigen::Map<tVectorXd, 0,
    // Eigen::InnerStride<3>> sol2(solution.data() + 2, size / 3);

    const tVectorXd &sol0_solved = solver.solve(res0);
    const tVectorXd &sol1_solved = solver.solve(res1);
    const tVectorXd &sol2_solved = solver.solve(res2);
    // cTimeUtil::Begin("assign");
    for (int i = 0; i < solution.size(); i++)
    {
        switch (i % 3)
        {
        case 0:
            solution[i] = sol0_solved[i / 3];
            break;
        case 1:
            solution[i] = sol1_solved[i / 3];
            break;
        case 2:
            solution[i] = sol2_solved[i / 3];
            break;

        default:
            SIM_ERROR("illegal case");
            break;
        }
    }
    // cTimeUtil::End("assign");
    // std::cout << "new sol0 = " << sol0_solved.transpose() << std::endl;
    // std::cout << "new res0 = " << res0.transpose() << std::endl;
    // std::cout << "new err = " << (A * sol0_solved - res0).transpose() <<
    // std::endl; std::cout << "map sol0 = " << sol0.transpose() << std::endl;
    // std::cout << "map res0 = " << res0.transpose() << std::endl;
    // std::cout << "map err = " << (A * sol0 - res0).transpose() << std::endl;
    // std::cout << "sol1 = " << sol1.transpose() << std::endl;
    // std::cout << "sol2 = " << sol2.transpose() << std::endl;
    // std::cout << "res = " << residual.transpose() << std::endl;
    // exit(0);
}
tVectorXd tmp;
tVectorXd cPDScene::CalcNextPosition() const
{
    // cTimeUtil::Begin("fast simulation calc next");
    // std::cout << "begin CalcNextPosition\n";
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
        // cTimeUtil::Begin("onestep");

        /*
            1. fixed x, calculate the d
            d = [d1, d2, ... dn]
            di = (x0^i - x1^i).normalized()
        */
        // std::cout << "step " << i << " X = " << Xnext.transpose() <<
        // std::endl;
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
        // cTimeUtil::Begin("calc_d");

        for (int j = 0; j < GetNumOfEdges(); j++)
        {
            int id0 = mEdgeArray[j]->mId0, id1 = mEdgeArray[j]->mId1;
            d.segment(j * 3, 3).noalias() =
                (Xnext.segment(3 * id0, 3) - Xnext.segment(3 * id1, 3))
                    .normalized() *
                mEdgeArray[j]->mRawLength;
        }
        // cTimeUtil::End("calc_d");
        // std::cout << "d = " << d.transpose() << std::endl;
        // std::cout << "J * d = " << (J * d).transpose() << std::endl;
        // std::cout << "b= " << b.transpose() << std::endl;
        /*
            2. fixed the d, calulcate the x
            x = (M + dt2 * L).inv() * (dt2 * J * d - b)
        */
        // // // cTimeUtil::BeginLazy("fast simulation sparse solve");
        // cTimeUtil::Begin("calc_res");
        tmp.noalias() =
            mInvMassMatrixDiag.cwiseProduct(dt2 * (J_sparse * d + fext)) + y;
        // cTimeUtil::End("calc_res");

        // 70 percent
        // cTimeUtil::Begin("solve");
        // Xnext.noalias() = I_plus_dt2_Minv_L_sparse_solver.solve(
        //     tmp);
        // cTimeUtil::End("solve");
        // tVectorXd sol;
        // std::cout << "A_fast = \n"
        //           << I_plus_dt2_Minv_L_sparse_fast << std::endl;
        // cTimeUtil::Begin("solve_fast");
        // printf("[debug] A size = %d, nonzeros %d\n", Xnext.size(),
        // I_plus_dt2_Minv_L_sparse.nonZeros());
        SolveFast(I_plus_dt2_Minv_L_sparse_fast,
                  I_plus_dt2_Minv_L_sparse_fast_solver, tmp, Xnext);
        // cTimeUtil::End("solve_fast");
        // std::cout << "new sol = " << sol.transpose() << std::endl;
        // std::cout << "old sol = " << Xnext.transpose() << std::endl;
        // std::cout << "diff = " << (sol - Xnext).transpose() << std::endl;
        // std::cout << "---------------\n";
        // std::cout << "A = \n"
        //           << I_plus_dt2_Minv_L_sparse << std::endl;

        // std::cout << "res = \n"
        //           << tmp.transpose() << std::endl;

        // exit(0);
        // // cTimeUtil::EndLazy("fast simulation sparse solve");
        //  = I_plus_dt2_Minv_L_inv * ();
        if (Xnext.hasNaN())
        {
            std::cout << "Xnext has Nan, exit = " << Xnext.transpose()
                      << std::endl;
            exit(0);
        }
        // cTimeUtil::End("onestep");
    }
    // cTimeUtil::ClearLazy("fast simulation sparse solve");
    // std::cout << "done, xnext = " << Xnext.transpose() << std::endl;
    // exit(0);
    // cTimeUtil::End("fast simulation calc next");
    return Xnext;
}

void cPDScene::UpdateSubstep()
{
    // cTimeUtil::Begin("substep");
    ClearForce();

    // cTimeUtil::Begin("substep_calc_next_pos");
    const tVectorXd &Xnext = CalcNextPosition();
    // cTimeUtil::End("substep_calc_next_pos");
    mXpre.noalias() = mXcur;
    mXcur.noalias() = Xnext;
    // cTimeUtil::Begin("substep_update_pos");
    UpdateCurNodalPosition(mXcur);
    // cTimeUtil::End("substep_update_pos");
    // cTimeUtil::End("substep");
}

void cPDScene::Reset() { cSimScene::Reset(); }

void cPDScene::Update(double dt) { cSimScene::Update(dt); }

double CalcTriangleSquare(const tVector &v0, const tVector &v1,
                          const tVector &v2)
{
    tVector e0 = v1 - v0, e1 = v2 - v0, e2 = v2 - v1;
    SIM_ASSERT(std::fabs(e0[3]) < 1e-10 && std::fabs(e1[3]) < 1e-10 &&
               std::fabs(e2[3]) < 1e-10);
    double s =
        e2.norm() / std::sin(std::acos(e0.dot(e1) / (e0.norm() * e1.norm())));
    SIM_ASSERT(std::isnan(s) == false);
    return s;
}
double CalcTriangleSquare(tTriangle *tri, std::vector<tVertex *> v_array)
{
    return CalcTriangleSquare(v_array[tri->mId0]->mPos,
                              v_array[tri->mId1]->mPos,
                              v_array[tri->mId2]->mPos);
}
/**
 * \biref           calculate the bending system matrix contribution triplet
 * According to the note, we only need to add some more entries into the system
 * matrix to support bending These triplets are calcualted here
 */
void cPDScene::AddBendTriplet(tEigenArr<tTriplet> &old_lst) const
{
    cTimeUtil::Begin("build bending triplet");
    printf("[debug] bending stiffness %.4f\n", mBendingStiffness);
    double h2_coef =
        mIdealDefaultTimestep * mIdealDefaultTimestep * mBendingStiffness;

    int num_of_dof = GetNumOfFreedom();
    // 1. dense implemention
    // tMatrixXd dense = tMatrixXd::Zero(num_of_dof, num_of_dof);
    // {
    //     for (int i = 0; i < GetNumOfEdges(); i++)
    //     {
    //         const auto &e = mEdgeArray[i];
    //         if (e->mIsBoundary == false)
    //         {
    //             int vid[4] = {e->mId0,
    //                           e->mId1,
    //                           SelectAnotherVerteix(mTriangleArray[e->mTriangleId0],
    //                           e->mId0, e->mId1),
    //                           SelectAnotherVerteix(mTriangleArray[e->mTriangleId1],
    //                           e->mId0, e->mId1)};
    //             // printf("[debug] bending, tri %d and tri %d, shared edge:
    //             %d, total vertices: %d %d %d %d\n",
    //             //        e->mTriangleId0, e->mTriangleId1, i, vid[0],
    //             vid[1], vid[2], vid[3]); tVector cot_vec =
    //             CalculateCotangentCoeff(
    //                 mVertexArray[vid[0]]->mPos,
    //                 mVertexArray[vid[1]]->mPos,
    //                 mVertexArray[vid[2]]->mPos,
    //                 mVertexArray[vid[3]]->mPos);
    //             tMatrixXd KLi = tMatrixXd::Zero(3, num_of_dof);
    //             for (int j = 0; j < 4; j++)
    //             {
    //                 KLi.block(0, 3 * vid[j], 3, 3).noalias() =
    //                 tMatrix3d::Identity() * cot_vec[j];
    //             }
    //             double s =
    //             CalcTriangleSquare(mTriangleArray[e->mTriangleId0],
    //             mVertexArray) +
    //                        CalcTriangleSquare(mTriangleArray[e->mTriangleId1],
    //                        mVertexArray);
    //             dense += s * KLi.transpose() * KLi;
    //         }
    //     }
    // }
    // dense = h2_coef * mInvMassMatrixDiag.asDiagonal().toDenseMatrix() *
    // dense; std::cout << "dense = \n"
    //           << dense << std::endl;
    tEigenArr<tTriplet> sparse_tri(0);
    // 2. sparse implemention
    {
        for (int i = 0; i < GetNumOfEdges(); i++)
        {
            const auto &e = mEdgeArray[i];
            if (e->mIsBoundary == false)
            {
                int vid[4] = {
                    e->mId0, e->mId1,
                    SelectAnotherVerteix(mTriangleArray[e->mTriangleId0],
                                         e->mId0, e->mId1),
                    SelectAnotherVerteix(mTriangleArray[e->mTriangleId1],
                                         e->mId0, e->mId1)};
                // printf("[debug] bending, tri %d and tri %d, shared edge: %d,
                // total vertices: %d %d %d %d\n",
                //        e->mTriangleId0, e->mTriangleId1, i, vid[0], vid[1],
                //        vid[2], vid[3]);
                tVector cot_vec = CalculateCotangentCoeff(
                    mVertexArray[vid[0]]->mPos, mVertexArray[vid[1]]->mPos,
                    mVertexArray[vid[2]]->mPos, mVertexArray[vid[3]]->mPos);
                double square =
                    CalcTriangleSquare(mTriangleArray[e->mTriangleId0],
                                       mVertexArray) +
                    CalcTriangleSquare(mTriangleArray[e->mTriangleId1],
                                       mVertexArray);
                for (int a = 0; a < 4; a++)
                    for (int b = 0; b < 4; b++)
                    {
                        for (int k = 0; k < 3; k++)
                        {
                            double value = cot_vec[a] * cot_vec[b] *
                                           mInvMassMatrixDiag[3 * vid[a] + k] *
                                           h2_coef * square;
                            sparse_tri.push_back(tTriplet(
                                3 * vid[a] + k, 3 * vid[b] + k, value));
                        }
                    }
                // tMatrixXd KLi = tMatrixXd::Zero(3, num_of_dof);
                // for (int j = 0; j < 4; j++)
                // {
                //     KLi.block(0, 3 * vid[j], 3, 3).noalias() =
                //     tMatrix3d::Identity() * cot_vec[j];
                // }

                // dense += s * KLi.transpose() * KLi;
            }
        }
    }

    // tSparseMat sparse(num_of_dof, num_of_dof);
    // sparse.setFromTriplets(sparse_tri.begin(), sparse_tri.end());
    // std::cout << "sparse = \n"
    //           << sparse << std::endl;
    // 3. compare test

    // {
    //     auto diff = dense - sparse;
    //     std::cout << "diff norm = " << diff.norm() << std::endl;
    // }
    // return sparse_tri;
    old_lst.insert(old_lst.begin(), sparse_tri.begin(), sparse_tri.end());
    cTimeUtil::End("build bending triplet");
    // SIM_ERROR("need to be impled");
}
