#pragma once
// #include <Eigen/Dense>
#include "DeviceManager.h"
#include <memory>

#include <k4a/k4a.h>

// typedef Eigen::MatrixXi tMatrixXi;
// typedef Eigen::MatrixXd tMatrixXd;
// typedef Eigen::Matrix3d tMatrix3d;
// typedef Eigen::VectorXd tVectorXd;

class cKinectManager : public cDeviceManager
{
public:
    cKinectManager();
    virtual ~cKinectManager();
    virtual tMatrixXi GetDepthImage();
    virtual double GetDepthUnit_mm();
    virtual tMatrixXi GetIrImage();
    virtual tMatrix3d GetDepthIntrinsicMtx() const;
    virtual tVectorXd GetDepthIntrinsicDistCoef() const;

protected:
    virtual void Init();
    k4a_device_t mDevice;
    virtual k4a_capture_t GetCapture() const;
    virtual k4a_calibration_camera_t GetDepthCalibration() const;
    // openni::VideoFrameRef m_depthFrame;
    // openni::Device m_device;
    // openni::VideoStream m_depthStream;
    // int m_width, m_height;
    // tMatrixXi depth_mat, ir_mat;
    // openni::VideoFrameRef m_irFrame;
    // openni::VideoStream m_irStream;
};