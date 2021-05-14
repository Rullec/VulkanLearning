#pragma once
#include "OpenNI.h"
#include "utils/MathUtil.h"
// #include Eigen/Dense"
// typedef Eigen::MatrixXd tMatrixXd;
// typedef Eigen::MatrixXi tMatrixXi;

class cDepthSampler
{
public:
    cDepthSampler();
    virtual tMatrixXi GetDepthImage();
    virtual double GetDepthUnit_mm();

protected:
    virtual void Init();
    openni::VideoFrameRef m_depthFrame;
    openni::Device m_device;
    openni::VideoStream m_depthStream;
    int m_width, m_height;
    tMatrixXi mat;
    // texture members
    // unsigned int m_nTexMapX;
    // unsigned int m_nTexMapY;
    // openni::RGB888Pixel *m_pTexMap;
};