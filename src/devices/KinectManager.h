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
    cKinectManager(std::string mode);
    virtual ~cKinectManager();
    virtual tMatrixXi GetDepthImage();
    virtual double GetDepthUnit_mm();
    virtual tMatrixXi GetIrImage();
    virtual std::string GetDepthMode() const;
    virtual void SetDepthMode(std::string mode);
    virtual tMatrix3d GetDepthIntrinsicMtx_sdk() const;
    virtual tVectorXd GetDepthIntrinsicDistCoef_sdk() const;
    virtual tMatrix3d GetColorIntrinsicMtx_sdk() const;
    virtual tVectorXd GetColorIntrinsicDistCoef_sdk() const;

    virtual tMatrix3d GetDepthIntrinsicMtx_self() const;
    virtual tVectorXd GetDepthIntrinsicDistCoef_self() const;
    virtual std::vector<tMatrixXi> GetColorImage() const;
    virtual tMatrixXi GetDepthToColorImage() const;

protected:
    virtual void Init();
    virtual void CloseDevice();
    k4a_device_t mDevice;
    k4a_device_configuration_t mConfig;
    k4a_depth_mode_t mDepthMode;
    virtual k4a_capture_t GetCapture() const;
    virtual k4a_calibration_camera_t GetDepthCalibration() const;
    virtual k4a_calibration_camera_t GetColorCalibration() const;
    static std::string BuildStringFromDepthMode(k4a_depth_mode_t mode);
    static k4a_depth_mode_t BuildModeFromString(std::string);

    // openni::VideoFrameRef m_depthFrame;
    // openni::Device m_device;
    // openni::VideoStream m_depthStream;
    // int m_width, m_height;
    // tMatrixXi depth_mat, ir_mat;
    // openni::VideoFrameRef m_irFrame;
    // openni::VideoStream m_irStream;
};