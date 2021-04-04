#include "MassSpringScene.h"
#include "utils/JsonUtil.h"
#include <iostream>
tSpring::tSpring()
{
    mRawLength = 0;
    mK = 0;
    mId0 = -1;
    mId1 = -1;
}

cMSScene::cMSScene()
{
    mSpringArray.clear();
}

cMSScene::~cMSScene()
{
    for (auto x : mSpringArray)
        delete x;
    mSpringArray.clear();
}

void cMSScene::Init(const std::string &conf_path)
{
    Json::Value root;
    cJsonUtil::LoadJson(conf_path, root);

    cSimScene::Init(conf_path);
    mMaxNewtonIters = cJsonUtil::ParseAsDouble("max_newton_iters", root);
}

void cMSScene::Update(double dt)
{
    cSimScene::Update(dt);
}

void cMSScene::Reset()
{
    cSimScene::Reset();
}

/**
 * \brief   discretazation from square cloth to mass spring system
 * 
*/
void cMSScene::InitGeometry()
{
    int spring_id = 0;
    int vertex_id = 0;
    int gap = mSubdivision + 1;
    double unit_edge_length = mClothWidth / mSubdivision;
    // for all row lines' edges
    for (int i = 0; i < mSubdivision + 1; i++)
    {
        for (int j = 0; j < mSubdivision; j++)
        {
            // if i is row index
            {
                tSpring *spr = new tSpring();
                spr->mRawLength = unit_edge_length;
                spr->mK = mStiffness;
                spr->mId0 = gap * i + j;
                spr->mId1 = gap * i + j + 1;
                mSpringArray.push_back(spr);
                printf("create spring %d between %d and %d\n",
                       mSpringArray.size() - 1, spr->mId0, spr->mId1);
            }
            // if i is column index
            {
                tSpring *spr = new tSpring();
                spr->mRawLength = unit_edge_length;
                spr->mK = mStiffness;
                spr->mId0 = gap * j + i;
                spr->mId1 = gap * (j + 1) + i;
                mSpringArray.push_back(spr);
                printf("create spring %d between %d and %d\n",
                       mSpringArray.size() - 1, spr->mId0, spr->mId1);
            }
        }
    }

    // set up the vertex pos data
    // in XOY plane
    {
        for (int i = 0; i < gap; i++)
            for (int j = 0; j < gap; j++)
            {
                tVertex *v = new tVertex();
                v->mMass = mClothMass / (gap * gap);
                v->mPos =
                    tVector(unit_edge_length * i, unit_edge_length * j, 0, 1);
                v->mColor = tVector(0, 196.0 / 255, 1, 0);
                mVertexArray.push_back(v);
                v->muv =
                    tVector2f(i * 1.0 / mSubdivision, j * 1.0 / mSubdivision);
                printf("create vertex %d at (%.3f, %.3f), uv (%.3f, %.3f)\n",
                       mVertexArray.size() - 1, v->mPos[0], v->mPos[1],
                       v->muv[0], v->muv[1]);
            }
    }

    // init the buffer
    {
        int num_of_square = mSubdivision * mSubdivision;
        int num_of_triangles = num_of_square * 2;
        int num_of_vertices = num_of_triangles * 3;
        int size_per_vertices = 8;
        mTriangleDrawBuffer.resize(num_of_vertices * size_per_vertices);
    }
    {
        int num_of_edges = 2 * (gap - 1) * gap;
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

void cMSScene::UpdateSubstep()
{
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
        mXnext = CalcNextPositionSemiImplicit();
        break;
    }
    case eIntegrationScheme::MS_IMPLICIT:
        mXnext = CalcNextPositionImplicit();
        break;
    default:
        SIM_ERROR("Unsupported integration scheme {}", mScheme);
        break;
    }
    // std::cout << "mXnext = " << mXnext.transpose() << std::endl;
    UpdatePreNodalPosition(mXcur);
    UpdateCurNodalPosition(mXnext);
}

/**
 * \brief            calculate inv mass mat
*/
void cMSScene::CalcInvMassMatrix() const {}

/**
 * \brief       external force
*/
extern const tVector gGravity;
void cMSScene::CalcExtForce(tVectorXd &ext_force) const
{
    // apply gravity
    for (int i = 0; i < mVertexArray.size(); i++)
    {
        ext_force.segment(3 * i, 3) +=
            gGravity.segment(0, 3) * mVertexArray[i]->mMass;
    }
}

/**
 * \brief       internal force
*/
void cMSScene::CalcIntForce(const tVectorXd &xcur, tVectorXd &int_force) const
{
    int id0, id1;
    double dist;
    tVector3d pos0, pos1;
    for (auto &spr : this->mSpringArray)
    {
        // 1. calcualte internal force for each spring
        id0 = spr->mId0;
        id1 = spr->mId1;
        pos0 = xcur.segment(id0 * 3, 3);
        pos1 = xcur.segment(id1 * 3, 3);
        dist = (pos0 - pos1).norm();
        tVector3d force0 = spr->mK * (spr->mRawLength - dist) *
                           (pos0 - pos1).segment(0, 3) / dist;
        tVector3d force1 = -force0;

        // 2. add force
        int_force.segment(3 * id0, 3) += force0.segment(0, 3);
        int_force.segment(3 * id1, 3) += force1.segment(0, 3);
    }
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
    tVectorXd next_pos =
        dt2 * mInvMassMatrixDiag.cwiseProduct(mIntForce + mExtForce + mDampingForce) +
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
        // next_pos.segment(i * 3, 3) = mXcur.segment(i * 3, 3);
        // std::cout << mXcur.segment(i * 3, 3).transpose() << std::endl;
    }
}

// -----------implicit methods-------------
/**
 * \brief           given current pos, calculate the value of dynamic equation
 *      G(x) = 2 * Xcur - Xpre - X + Minv * dt2 * (Fext + Fint) = 0
*/
void cMSScene::CalcGxImplicit(const tVectorXd &x, tVectorXd &Gx,
                              tVectorXd &fint_buffer,
                              tVectorXd &fext_buffer,
                              tVectorXd &damp_buffer) const
{
    CalcIntForce(x, fint_buffer);
    CalcExtForce(fext_buffer);
    CalcDampingForce((mXcur - mXpre) / mCurdt, damp_buffer);
    Gx.noalias() =
        2 * mXcur - mXpre - x +
        mCurdt * mCurdt *
            mInvMassMatrixDiag.cwiseProduct(fext_buffer + fint_buffer + damp_buffer);
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
    for (auto &spr : this->mSpringArray)
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
            spr->mK * I3 -
            spr->mK * spr->mRawLength *
                (I3 * dist - (pos0 - pos1) * (pos0 - pos1).transpose() / dist) /
                (dist * dist);

        if (dfidxi.hasNaN() == true)
        {
            printf("df%ddx%d has Nan, id0 = %d, id1 = %d, dist = %.5f\n", id0,
                   id0, id0, id1, dist);
            std::cout << "dfidxi = \n"
                      << dfidxi << std::endl;
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

typedef Eigen::Triplet<double> tTriplet;
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
    for (auto &spr : this->mSpringArray)
    {
        id0 = spr->mId0;
        id1 = spr->mId1;

        pos0 = x.segment(3 * id0, 3);
        pos1 = x.segment(3 * id1, 3);
        double dist = (pos0 - pos1).norm();
        const tMatrix3d &I3 = tMatrix3d::Identity(3, 3);
        tMatrix3d dfidxi =
            spr->mK * I3 -
            spr->mK * spr->mRawLength *
                (I3 * dist - (pos0 - pos1) * (pos0 - pos1).transpose() / dist) /
                (dist * dist);

        if (dfidxi.hasNaN() == true)
        {
            printf("df%ddx%d has Nan, id0 = %d, id1 = %d, dist = %.5f\n", id0,
                   id0, id0, id1, dist);
            std::cout << "dfidxi = \n"
                      << dfidxi << std::endl;
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