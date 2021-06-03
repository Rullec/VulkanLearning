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
    virtual tMatrix3d GetDepthIntrinsicMtx_sdk() const = 0;
    virtual tVectorXd GetDepthIntrinsicDistCoef_sdk() const = 0;
    virtual tMatrix3d GetDepthIntrinsicMtx_self() const = 0;
    virtual tVectorXd GetDepthIntrinsicDistCoef_self() const = 0;

protected:
};
