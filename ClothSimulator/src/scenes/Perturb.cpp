#include "Perturb.h"
#include "utils/LogUtil.h"

tPerturb::tPerturb()
{
    mTriangle = nullptr;
    mBarycentricCoords.setZero();
}

tMatrixXd tPerturb::CalcForceOnEachVertex() const
{
    SIM_ERROR("hasn't been impled yet");
    exit(0);
}
