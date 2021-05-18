#pragma once
#include "OpenNI.h"
#include "utils/MathUtil.h"
// #include Eigen/Dense"
// typedef Eigen::MatrixXd tMatrixXd;
// typedef Eigen::MatrixXi tMatrixXi;

class cVideoManager
{
public:
    cVideoManager();
    virtual tMatrixXi GetDepthImage();
    virtual double GetDepthUnit_mm();
    virtual tMatrixXi GetIrImage();

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