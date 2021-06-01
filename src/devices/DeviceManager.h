#pragma once
#include "utils/MathUtil.h"

class cDeviceManager
{
public:
    cDeviceManager();
    virtual ~cDeviceManager() = 0;
    virtual tMatrixXi GetDepthImage() = 0;
    virtual double GetDepthUnit_mm() = 0;
    virtual tMatrixXi GetIrImage() = 0;
    virtual tMatrix3d GetDepthIntrinsicMtx() const = 0;
    virtual tVectorXd GetDepthIntrinsicDistCoef() const = 0;

protected:
};
