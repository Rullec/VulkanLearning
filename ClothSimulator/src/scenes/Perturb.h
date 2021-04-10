#pragma once
#include "utils/MathUtil.h"

/**
 * \brief           user perturb force
*/
struct tTriangle;
struct tPerturb
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    tPerturb();
    tMatrixXd CalcForceOnEachVertex() const;

    tTriangle *mTriangle; // the affected triangle
    tVector3d
        mBarycentricCoords; // the barycentric coordinates of raw raycast point on the affected triangle
};