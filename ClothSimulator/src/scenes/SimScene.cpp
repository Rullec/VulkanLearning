#include "SimScene.h"
#include "utils/JsonUtil.h"
#include <iostream>

std::string gIntegrationSchemeStr
    [cSimScene::eIntegrationScheme::NUM_OF_INTEGRATION_SCHEMES] = {
        "semi_implicit", "implicit"};

cSimScene::eIntegrationScheme BuildIntegrationScheme(const std::string &str)
{
    int i = 0;
    for (i = 0; i < cSimScene::eIntegrationScheme::NUM_OF_INTEGRATION_SCHEMES;
         i++)
    {
        if (str == gIntegrationSchemeStr[i])
        {
            break;
        }
    }

    SIM_ASSERT(i != cSimScene::eIntegrationScheme::NUM_OF_INTEGRATION_SCHEMES);
    return static_cast<cSimScene::eIntegrationScheme>(i);
}

tVertex::tVertex()
{
    mMass = 0;
    mPos.setZero();
    mColor = tVector::Ones();
}

tSpring::tSpring()
{
    mRawLength = 0;
    mK = 0;
    mId0 = -1;
    mId1 = -1;
}

cSimScene::cSimScene()
{
    mVertexArray.clear();
    mSpringArray.clear();
}

void cSimScene::Init(const std::string &conf_path)
{
    // 1. load config
    Json::Value root;
    cJsonUtil::LoadJson(conf_path, root);

    mClothWidth = cJsonUtil::ParseAsDouble("cloth_size", root);
    mClothMass = cJsonUtil::ParseAsDouble("cloth_mass", root);
    mSubdivision = cJsonUtil::ParseAsInt("subdivision", root);
    mStiffness = cJsonUtil::ParseAsDouble("stiffness", root);
    mDamping = cJsonUtil::ParseAsDouble("damping", root);
    mMaxNewtonIters = cJsonUtil::ParseAsDouble("max_newton_iters", root);
    mScheme = BuildIntegrationScheme(
        cJsonUtil::ParseAsString("integration_scheme", root));
    SIM_INFO("cloth total width {} subdivision {} K {}", mClothWidth,
             mSubdivision, mStiffness);
    // 2. create geometry, dot allocation
    InitGeometry();
    InitConstraint(root);

    // 3. set up the init pos
    CalcNodePositionVector(mXpre);
    mXcur.noalias() = mXpre;

    std::cout << "init sim scene done\n";
}

/**
 * \brief           Update the simulation procedure
*/
void cSimScene::Update(double delta_time)
{
    double default_dt = 1e-3;
    if (delta_time < default_dt)
        default_dt = delta_time;
    printf("[debug] sim scene update cur time = %.4f\n", mCurTime);
    while (delta_time > 1e-7)
    {
        if (delta_time < default_dt)
            default_dt = delta_time;
        cScene::Update(default_dt);

        // std::cout << "[update] x = " << mXcur.transpose() << std::endl;
        // 1. clear force
        ClearForce();
        // 2. calculate force
        CalcIntForce(mXcur, mIntForce);
        CalcExtForce(mExtForce);
        CalcDampingForce((mXcur - mXpre) / default_dt, mDampingForce);
        // std::cout << "mInt force = " << mIntForce.transpose() << std::endl;
        // std::cout << "mExt force = " << mExtForce.transpose() << std::endl;
        // 3. forward simulation
        tVectorXd mXnext = tVectorXd::Zero(GetNumOfFreedom());
        switch (mScheme)
        {
        case eIntegrationScheme::SEMI_IMPLICIT:
            mXnext = CalcNextPositionSemiImplicit();
            break;
        case eIntegrationScheme::IMPLICIT:
            mXnext = CalcNextPositionImplicit();
            break;
        default:
            SIM_ERROR("Unsupported integration scheme {}", mScheme);
            break;
        }
        // std::cout << "mXnext = " << mXnext.transpose() << std::endl;
        UpdatePreNodalPosition(mXcur);
        UpdateCurNodalPosition(mXnext);

        delta_time -= default_dt;
    }

    // 4. post process
    CalcTriangleDrawBuffer();
    CalcEdgesDrawBuffer();
    // std::cout << "xcur = " << mXcur.transpose() << std::endl;
    SIM_ASSERT(mXcur.hasNaN() == false);
}

/**
 * \brief           Reset the whole scene
*/
void cSimScene::Reset()
{
    cSimScene::Reset();
    ClearForce();
}

/**
 * \brief           Get number of vertices
*/
int cSimScene::GetNumOfVertices() const { return mVertexArray.size(); }

/**
 * \brief   discretazation from square cloth to mass spring system
 * 
*/
void cSimScene::InitGeometry()
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

/**
 * \brief       clear all forces
*/
void cSimScene::ClearForce()
{
    int dof = GetNumOfFreedom();
    mIntForce.noalias() = tVectorXd::Zero(dof);
    mExtForce.noalias() = tVectorXd::Zero(dof);
}

/**
 * \brief            calculate inv mass mat
*/
void cSimScene::CalcInvMassMatrix() const {}

/**
 * \brief       external force
*/
extern const tVector gGravity;
void cSimScene::CalcExtForce(tVectorXd &ext_force) const
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
void cSimScene::CalcIntForce(const tVectorXd &xcur, tVectorXd &int_force) const
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
tVectorXd cSimScene::CalcNextPositionSemiImplicit() const
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
tVectorXd cSimScene::CalcNextPositionImplicit()
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

void cSimScene::UpdateCurNodalPosition(const tVectorXd &newpos)
{
    mXcur = newpos;
    for (int i = 0; i < mVertexArray.size(); i++)
    {
        mVertexArray[i]->mPos.segment(0, 3).noalias() = mXcur.segment(i * 3, 3);
    }
}

void cSimScene::UpdatePreNodalPosition(const tVectorXd &xpre) { mXpre = xpre; }
void cSimScene::GetVertexRenderingData() {}

int cSimScene::GetNumOfFreedom() const { return GetNumOfVertices() * 3; }

void CalcTriangleDrawBufferSingle(tVertex *v0, tVertex *v1, tVertex *v2,
                                  tVectorXf &buffer, int &st_pos)
{
    // std::cout << "buffer size " << buffer.size() << " st pos " << st_pos << std::endl;
    buffer.segment(st_pos, 3) = v0->mPos.segment(0, 3).cast<float>();
    buffer.segment(st_pos + 3, 3) = v0->mColor.segment(0, 3).cast<float>();
    st_pos += 8;
    buffer.segment(st_pos, 3) = v1->mPos.segment(0, 3).cast<float>();
    buffer.segment(st_pos + 3, 3) = v1->mColor.segment(0, 3).cast<float>();
    st_pos += 8;
    buffer.segment(st_pos, 3) = v2->mPos.segment(0, 3).cast<float>();
    buffer.segment(st_pos + 3, 3) = v2->mColor.segment(0, 3).cast<float>();
    st_pos += 8;
}
void CalcEdgeDrawBufferSingle(tVertex *v0, tVertex *v1, tVectorXf &buffer,
                              int &st_pos)
{

    buffer.segment(st_pos, 3) = v0->mPos.segment(0, 3).cast<float>();
    buffer.segment(st_pos + 3, 3) = tVector3f(0, 0, 0);
    st_pos += 8;
    buffer.segment(st_pos, 3) = v1->mPos.segment(0, 3).cast<float>();
    buffer.segment(st_pos + 3, 3) = tVector3f(0, 0, 0);
    st_pos += 8;
}

const tVectorXf &cSimScene::GetTriangleDrawBuffer()
{
    return mTriangleDrawBuffer;
}
/**
 * \brief           Calculate vertex rendering data
*/
void cSimScene::CalcTriangleDrawBuffer()
{
    mTriangleDrawBuffer.fill(std::nan(""));
    // counter clockwise
    int gap = mSubdivision + 1;
    int st = 0;
    for (int i = 0; i < mSubdivision; i++)     // row
        for (int j = 0; j < mSubdivision; j++) // column
        {
            // left up coner
            int left_up = gap * i + j;
            int right_up = left_up + 1;
            int left_down = left_up + gap;
            int right_down = right_up + gap;
            // mVertexArray[left_up]->mPos *= (1 + 1e-3);
            CalcTriangleDrawBufferSingle(
                mVertexArray[right_down], mVertexArray[left_up],
                mVertexArray[left_down], mTriangleDrawBuffer, st);
            CalcTriangleDrawBufferSingle(
                mVertexArray[right_down], mVertexArray[right_up],
                mVertexArray[left_up], mTriangleDrawBuffer, st);
        }
}

const tVectorXf &cSimScene::GetEdgesDrawBuffer() { return mEdgesDrawBuffer; }

void cSimScene::CalcEdgesDrawBuffer()
{
    mEdgesDrawBuffer.fill(std::nan(""));
    int st = 0;
    int gap = mSubdivision + 1;

    // for all row lines' edges
    for (int i = 0; i < mSubdivision + 1; i++)
    {
        for (int j = 0; j < mSubdivision; j++)
        {
            // printf("[debug] edge from %d to %d: st %d / %d\n", i, j, st, mEdgesDrawBuffer.size());
            // if i is row index
            {
                int Id0 = gap * i + j;
                int Id1 = gap * i + j + 1;
                CalcEdgeDrawBufferSingle(mVertexArray[Id0], mVertexArray[Id1],
                                         mEdgesDrawBuffer, st);
            }
            // if i is column index
            {

                int Id0 = gap * j + i;
                int Id1 = gap * (j + 1) + i;
                CalcEdgeDrawBufferSingle(mVertexArray[Id0], mVertexArray[Id1],
                                         mEdgesDrawBuffer, st);
            }
        }
    }
}

void cSimScene::CalcNodePositionVector(tVectorXd &pos) const
{
    if (pos.size() != GetNumOfFreedom())
    {
        pos.noalias() = tVectorXd::Zero(GetNumOfFreedom());
    }
    for (int i = 0; i < mVertexArray.size(); i++)
    {
        pos.segment(i * 3, 3) = mVertexArray[i]->mPos.segment(0, 3);
    }
}

void cSimScene::InitConstraint(const Json::Value &root)
{
    if (root.isMember("constraints") == false)
        return;
    auto cons = cJsonUtil::ParseAsValue("constraints", root);
    if (cons.isMember("fixed_point") == true)
    {
        // 1. read all 2d constraint for fixed point
        auto fixed_cons = cJsonUtil::ParseAsValue("fixed_point", cons);
        int num_of_fixed_pts = fixed_cons.size();
        tEigenArr<tVector2f> fixed_tex_coords(num_of_fixed_pts);
        for (int i = 0; i < num_of_fixed_pts; i++)
        {
            fixed_tex_coords[i] = tVector2f(fixed_cons[i][0].asDouble(),
                                            fixed_cons[i][1].asDouble());
        }

        // 2. iterate over all vertices to find which point should be finally fixed
        mFixedPointIds.resize(num_of_fixed_pts, -1);
        std::vector<double> SelectedFixedPointApproxDist(num_of_fixed_pts,
                                                         std::nan(""));

        for (int v_id = 0; v_id < GetNumOfVertices(); v_id++)
        {
            const tVector2f &v_uv = mVertexArray[v_id]->muv;
            for (int j = 0; j < num_of_fixed_pts; j++)
            {
                double dist = (v_uv - fixed_tex_coords[j]).norm();
                if (std::isnan(SelectedFixedPointApproxDist[j]) ||
                    dist < SelectedFixedPointApproxDist[j])
                {
                    mFixedPointIds[j] = v_id;
                    SelectedFixedPointApproxDist[j] = dist;
                }
            }
        }

        // output
        for (int i = 0; i < num_of_fixed_pts; i++)
        {
            printf("[debug] fixed uv (%.3f %.3f) selected v_id %d uv (%.3f, "
                   "%.3f)\n",
                   fixed_tex_coords[i][0], fixed_tex_coords[i][1],
                   mFixedPointIds[i], mVertexArray[mFixedPointIds[i]]->muv[0],
                   mVertexArray[mFixedPointIds[i]]->muv[1]);
        }
    }

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
void cSimScene::CalcGxImplicit(const tVectorXd &x, tVectorXd &Gx,
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
void cSimScene::CalcdGxdxImplicit(const tVectorXd &x, tMatrixXd &dGdx) const
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
void cSimScene::TestdGxdxImplicit(const tVectorXd &x0, const tMatrixXd &Gx_ana)
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
void cSimScene::CalcdGxdxImplicitSparse(const tVectorXd &x,
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
void cSimScene::PushState(const std::string &name) const {}

void cSimScene::PopState(const std::string &name) {}

cSimScene::~cSimScene()
{
    for (auto x : mVertexArray)
        delete x;
    for (auto x : mSpringArray)
        delete x;

    mVertexArray.clear();
    mSpringArray.clear();
}

/**
 * \brief           add damping forces
*/
void cSimScene::CalcDampingForce(const tVectorXd &vel, tVectorXd &damping) const
{
    damping.noalias() = -vel * this->mDamping;
}