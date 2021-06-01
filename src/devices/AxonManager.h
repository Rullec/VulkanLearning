#pragma once

#include "DeviceManager.h"
#include "OpenNI.h"

// class cDeviceManager :
class cAxonManager : public cDeviceManager
{
public:
    cAxonManager();
    virtual tMatrixXi GetDepthImage();
    virtual double GetDepthUnit_mm();
    virtual tMatrixXi GetIrImage();
    virtual tMatrix3d GetDepthIntrinsicMtx() const;
    virtual tVectorXd GetDepthIntrinsicDistCoef() const;

protected:
    virtual void Init();
    openni::VideoFrameRef m_depthFrame;
    openni::Device m_device;
    openni::VideoStream m_depthStream;
    int m_width, m_height;
    tMatrixXi depth_mat, ir_mat;
    openni::VideoFrameRef m_irFrame;
    openni::VideoStream m_irStream;
};