#include "BaseCloth.h"
#include "geometries/Triangulator.h"
#include "utils/DefUtil.h"
#include "utils/JsonUtil.h"
#include "utils/LogUtil.h"

const std::string gClothTypeStr[eClothType::NUM_OF_CLOTH_TYPE] = {
    "semi_implicit", "implicit", "pbd", "pd", "linctex", "empty", "fem"};
cBaseCloth::cBaseCloth(eClothType cloth_type, int id_)
    : cBaseObject(eObjectType::CLOTH_TYPE, id_), mClothType(cloth_type)
{
    mTriangleArray.clear();
    mEdgeArray.clear();
    mVertexArray.clear();
    mFixedPointIds.clear();
}
cBaseCloth::~cBaseCloth() {}

void cBaseCloth::Init(const Json::Value &conf)
{
    mGeometryType =
        cJsonUtil::ParseAsString(cTriangulator::GEOMETRY_TYPE_KEY, conf);
    mDamping = cJsonUtil::ParseAsDouble(cBaseCloth::DAMPING_KEY, conf);
    mIdealDefaultTimestep =
        cJsonUtil::ParseAsDouble(cBaseCloth::DEFAULT_TIMESTEP_KEY, conf);
    InitGeometry(conf);
    InitConstraint(conf);
}

void cBaseCloth::Reset() { SetPos(mClothInitPos); }
void CalcTriangleDrawBufferSingle(tVertex *v0, tVertex *v1, tVertex *v2,
                                  Eigen::Map<tVectorXf> &buffer, int &st_pos)
{
    // std::cout << "buffer size " << buffer.size() << " st pos " << st_pos <<
    // std::endl;
    buffer.segment(st_pos, 3) = v0->mPos.segment(0, 3).cast<float>();
    buffer.segment(st_pos + 3, 3) = v0->mColor.segment(0, 3).cast<float>();
    st_pos += RENDERING_SIZE_PER_VERTICE;
    buffer.segment(st_pos, 3) = v1->mPos.segment(0, 3).cast<float>();
    buffer.segment(st_pos + 3, 3) = v1->mColor.segment(0, 3).cast<float>();
    st_pos += RENDERING_SIZE_PER_VERTICE;
    buffer.segment(st_pos, 3) = v2->mPos.segment(0, 3).cast<float>();
    buffer.segment(st_pos + 3, 3) = v2->mColor.segment(0, 3).cast<float>();
    st_pos += RENDERING_SIZE_PER_VERTICE;
}
void CalcEdgeDrawBufferSingle(tVertex *v0, tVertex *v1,
                              Eigen::Map<tVectorXf> &buffer, int &st_pos)
{

    buffer.segment(st_pos, 3) = v0->mPos.segment(0, 3).cast<float>();
    buffer.segment(st_pos + 3, 3) = tVector3f(0, 0, 0);
    st_pos += RENDERING_SIZE_PER_VERTICE;
    buffer.segment(st_pos, 3) = v1->mPos.segment(0, 3).cast<float>();
    buffer.segment(st_pos + 3, 3) = tVector3f(0, 0, 0);
    st_pos += RENDERING_SIZE_PER_VERTICE;
}

void CalcEdgeDrawBufferSingle(const tVector &v0, const tVector &v1,
                              Eigen::Map<tVectorXf> &buffer, int &st_pos)
{
    buffer.segment(st_pos, 3) = v0.segment(0, 3).cast<float>();
    buffer.segment(st_pos + 3, 3) = tVector3f(0, 0, 0);
    st_pos += RENDERING_SIZE_PER_VERTICE;
    buffer.segment(st_pos, 3) = v1.segment(0, 3).cast<float>();
    buffer.segment(st_pos + 3, 3) = tVector3f(1, 0, 0);
    st_pos += RENDERING_SIZE_PER_VERTICE;
}

void cBaseCloth::CalcTriangleDrawBuffer(Eigen::Map<tVectorXf> &res,
                                        int &st) const
{
    for (auto &tri : mTriangleArray)
    {

        CalcTriangleDrawBufferSingle(this->mVertexArray[tri->mId0],
                                     this->mVertexArray[tri->mId1],
                                     this->mVertexArray[tri->mId2], res, st);
    }
}

/**
 * \brief           calculate edge draw buffer
 */
void cBaseCloth::CalcEdgeDrawBuffer(Eigen::Map<tVectorXf> &res, int &st) const
{
    for (auto &e : mEdgeArray)
    {
        CalcEdgeDrawBufferSingle(mVertexArray[e->mId0], mVertexArray[e->mId1],
                                 res, st);
    }
}

void cBaseCloth::ClearForce()
{
    int dof = GetNumOfFreedom();
    mIntForce.noalias() = tVectorXd::Zero(dof);
    mExtForce.noalias() = tVectorXd::Zero(dof);
    mDampingForce.noalias() = tVectorXd::Zero(dof);
}
#include "scenes/Perturb.h"

void cBaseCloth::ApplyPerturb(tPerturb *pert)
{
    if (pert == nullptr)
        return;
    tVector force = pert->GetPerturbForce();
    mExtForce.segment(mTriangleArray[pert->mAffectedTriId]->mId0 * 3, 3) +=
        force.segment(0, 3) / 3;
    mExtForce.segment(mTriangleArray[pert->mAffectedTriId]->mId1 * 3, 3) +=
        force.segment(0, 3) / 3;
    mExtForce.segment(mTriangleArray[pert->mAffectedTriId]->mId2 * 3, 3) +=
        force.segment(0, 3) / 3;
}
/**
 * \brief           add damping forces
 */
void cBaseCloth::CalcDampingForce(const tVectorXd &vel,
                                  tVectorXd &damping) const
{
    damping.noalias() = -vel * mDamping;
}

void cBaseCloth::SetPos(const tVectorXd &newpos)
{
    mXcur = newpos;
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < mVertexArray.size(); i++)
    {
        mVertexArray[i]->mPos.segment(0, 3).noalias() = mXcur.segment(i * 3, 3);
    }
}

const tVectorXd &cBaseCloth::GetPos() const { return this->mXcur; }
void cBaseCloth::CalcExtForce(tVectorXd &ext_force) const
{
    // 1. apply gravity
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < mVertexArray.size(); i++)
    {
        ext_force.segment(3 * i, 3) +=
            gGravity.segment(0, 3) * mVertexArray[i]->mMass;
    }

    // std::cout << "add ext noise\n";
    // ext_force.segment(3 * (mVertexArray.size() - 1), 3) += tVector3d(0, 0,
    // 10);

    //  2. add perturb force
    // if (mPerturb != nullptr)
    // {
    //     tVector perturb_force = mPerturb->GetPerturbForce();
    //     // printf(
    //     //     "[debug] perturb vid %d %d %d, ",
    //     //     mPerturb->mAffectedVerticesId[0],
    //     //     mPerturb->mAffectedVerticesId[1], mPerturb        // std::cout
    //     <<
    //     //     "perturb force = " << perturb_force.transpose()
    //     //           << std::endl;->mAffectedVerticesId[2]);

    //     ext_force.segment(mPerturb->mAffectedVerticesId[0] * 3, 3) +=
    //         perturb_force.segment(0, 3) / 3;
    //     ext_force.segment(mPerturb->mAffectedVerticesId[1] * 3, 3) +=
    //         perturb_force.segment(0, 3) / 3;
    //     ext_force.segment(mPerturb->mAffectedVerticesId[2] * 3, 3) +=
    //         perturb_force.segment(0, 3) / 3;
    //     // 2. give the ray to the perturb, calculate force on each vertices
    //     // 3. apply the force
    // }
}

void cBaseCloth::InitConstraint(const Json::Value &root)
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
            SIM_ASSERT(fixed_cons[i].size() == 2);
            fixed_tex_coords[i] = tVector2f(fixed_cons[i][0].asDouble(),
                                            fixed_cons[i][1].asDouble());
        }

        // 2. iterate over all vertices to find which point should be finally
        // fixed
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
    for (auto &i : mFixedPointIds)
    {
        mInvMassMatrixDiag.segment(i * 3, 3).setZero();
        // printf("[debug] fixed point id %d at ", i);
        // exit(0);
        // next_pos.segment(i * 3, 3) = mXcur.segment(i * 3, 3);
        // std::cout << mXcur.segment(i * 3, 3).transpose() << std::endl;
    }
}

void cBaseCloth::InitGeometry(const Json::Value &conf)
{
    // 1. build the geometry
    mClothWidth = cJsonUtil::ParseAsDouble("cloth_size", conf);
    mClothMass = cJsonUtil::ParseAsDouble("cloth_mass", conf);

    cTriangulator::BuildGeometry(conf, mVertexArray, mEdgeArray,
                                 mTriangleArray);

    CalcNodePositionVector(mClothInitPos);

    // init the inv mass vector
    mInvMassMatrixDiag.noalias() = tVectorXd::Zero(GetNumOfFreedom());
    for (int i = 0; i < mVertexArray.size(); i++)
    {
        mInvMassMatrixDiag.segment(i * 3, 3).fill(1.0 / mVertexArray[i]->mMass);
    }
}

void cBaseCloth::CalcNodePositionVector(tVectorXd &pos) const
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

/**
 * \brief       internal force
 */
void cBaseCloth::CalcIntForce(const tVectorXd &xcur, tVectorXd &int_force) const
{
    // std::vector<std::atomic<double>> int_force_atomic(int_force.size());
    // for (int i = 0; i < int_force.size(); i++)
    //     int_force_atomic[i] = 0;
    // double res = 1;
    // std::vector<double> int_force_atomic(int_force.size());

    // std::cout << "input fint = " << int_force.transpose() << std::endl;
    int id0, id1;
    double dist;
#ifdef USE_OPENMP
#pragma omp parallel for private(id0, id1, dist)
#endif
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
        //         std::cout << "spring " << i << " force = " <<
        //         force0.transpose() << ", dist " << dist << ", v0 " << id0 <<
        //         " v1 " << id1 << std::endl;
        // std::cout << "spring " << i << ", v0 = " << id0 << " v1 = " << id1 <<
        // std::endl;
        // 2. add force
        {
#ifdef USE_OPENMP
#pragma omp atomic
#endif
            int_force[3 * id0 + 0] += force0[0];
#ifdef USE_OPENMP
#pragma omp atomic
#endif
            int_force[3 * id0 + 1] += force0[1];
#ifdef USE_OPENMP
#pragma omp atomic
#endif
            int_force[3 * id0 + 2] += force0[2];
#ifdef USE_OPENMP
#pragma omp atomic
#endif
            int_force[3 * id1 + 0] += -force0[0];
#ifdef USE_OPENMP
#pragma omp atomic
#endif
            int_force[3 * id1 + 1] += -force0[1];
#ifdef USE_OPENMP
#pragma omp atomic
#endif
            int_force[3 * id1 + 2] += -force0[2];
        }
    }
    // std::cout << "output fint = " << int_force.transpose() << std::endl;
    // exit(0);
}

int cBaseCloth::GetNumOfFreedom() const { return 3 * GetNumOfVertices(); }

eClothType cBaseCloth::BuildClothType(std::string str)
{
    for (int i = 0; i < eClothType::NUM_OF_CLOTH_TYPE; i++)
    {
        if (gClothTypeStr[i] == str)
        {
            return static_cast<eClothType>(i);
        }
    }
    SIM_ERROR("unsupported cloth type {}", str);
    return eClothType::NUM_OF_CLOTH_TYPE;
}

void cBaseCloth::SetCollisionDetecter(cCollisionDetecterPtr ptr)
{
    mColDetecter = ptr;
}