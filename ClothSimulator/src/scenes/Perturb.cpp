#include "Perturb.h"
#include "geometries/Primitives.h"
#include "utils/LogUtil.h"
#include <iostream>
tPerturb::tPerturb()
{
    mPerturbForce.setZero();
    for (int i = 0; i < 3; i++)
    {

        mAffectedVertices[i] = nullptr;
        mAffectedVerticesId[i] = -1;
    }
    mBarycentricCoords.setZero();
    mRectTri0 = mRectTri1 = nullptr;
    for (int i = 0; i < 4; i++)
        mRectVertices[i] = nullptr;
}

void tPerturb::InitTangentRect(const tVector &plane_normal)
{
    // 1. calculate center pos
    tVector center_pos = CalcPerturbPos();
    // std::cout << "center pos = " << center_pos.transpose() << std::endl;
    // 2. calculate rectangle vertices
    // each row is a vector
    tMatrixXd four_vectors = cMathUtil::ExpandFrictionCone(4, plane_normal);
    // std::cout << "four vecs = \n" << four_vectors << std::endl;
    double plane_scale = 1e10;
    four_vectors *= plane_scale;
    for (int i = 0; i < 4; i++)
    {
        // std::cout << 1 << std::endl;
        four_vectors.row(i) += center_pos;
        // std::cout << 2 << std::endl;
        mRectVertices[i] = new tVertex();
        // std::cout << 3 << std::endl;
        mRectVertices[i]->mPos = four_vectors.row(i);
        // std::cout << 4 << std::endl;
        // std::cout << "rect v" << i << " = "
        //           << mRectVertices[i]->mPos.transpose() << std::endl;
    }
}

void tPerturb::UpdatePerturb(const tVector &cur_camera_pos, const tVector &dir)
{
    tVector goal_pos =
        cMathUtil::RayCast(cur_camera_pos, dir, mRectVertices[0]->mPos,
                           mRectVertices[1]->mPos, mRectVertices[2]->mPos);
    if (goal_pos.hasNaN() == true)
    {
        goal_pos.noalias() =
            cMathUtil::RayCast(cur_camera_pos, dir, mRectVertices[0]->mPos,
                               mRectVertices[2]->mPos, mRectVertices[3]->mPos);
        if (goal_pos.hasNaN() == true)
        {
            SIM_ERROR("ray has no intersection with the rect range, please "
                      "scale the plane");
            exit(0);
        }
    }

    tVector cur_perturb_pos = CalcPerturbPos();
    mPerturbForce = (goal_pos - cur_perturb_pos) * 10;
    // std::cout << "[update] cur perturb pos = " << cur_perturb_pos.transpose()
    //           << " goal pos = " << goal_pos.transpose() << std::endl;
}

tVector tPerturb::CalcPerturbPos() const
{
    tVector center_pos = tVector::Zero();

    for (int i = 0; i < 3; i++)
    {
        center_pos += mAffectedVertices[i]->mPos * mBarycentricCoords[i];
    }
    center_pos[3] = 1;
    return center_pos;
}

tVector tPerturb::GetPerturbForce() const { return mPerturbForce; }