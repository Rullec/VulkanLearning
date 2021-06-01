#include "AxonManager.h"
// #include "Viewer.h"
#include "AXonLink.h"

int InitDevice(openni::Status &rc, openni::Device &device,
               openni::VideoStream &depth, openni::VideoStream &color,
               openni::VideoStream &ir

)
{

    int nResolutionColor = 0;
    int nResolutionDepth = 0;
    int lastResolutionX = 0;
    int lastResolutionY = 0;
    const char *deviceURI = openni::ANY_DEVICE;
    // if (argc > 1)
    // {
    // 	deviceURI = argv[1];
    // }

    rc = openni::OpenNI::initialize();

    // printf("After initialization:\n%s\n",
    // openni::OpenNI::getExtendedError());
    printf("[debug] begin to init axon depth camera\n");

    rc = device.open(deviceURI);
    if (rc != openni::STATUS_OK)
    {
        printf("SimpleViewer: Device open failed:\n%s\n",
               openni::OpenNI::getExtendedError());
        openni::OpenNI::shutdown();
        return 1;
    }
    const openni::SensorInfo *info = device.getSensorInfo(openni::SENSOR_COLOR);
    // if (info)
    // {
    //     for (int i = 0; i < info->getSupportedVideoModes().getSize(); i++)
    //     {
    //         printf("Color info video %d %dx%d FPS %d f %d\n", i,
    //                info->getSupportedVideoModes()[i].getResolutionX(),
    //                info->getSupportedVideoModes()[i].getResolutionY(),
    //                info->getSupportedVideoModes()[i].getFps(),
    //                info->getSupportedVideoModes()[i].getPixelFormat());
    //         if ((info->getSupportedVideoModes()[i].getResolutionX() !=
    //         lastResolutionX) ||
    //         (info->getSupportedVideoModes()[i].getResolutionY() !=
    //         lastResolutionY))
    //         {
    //             nResolutionColor++;
    //             lastResolutionX =
    //             info->getSupportedVideoModes()[i].getResolutionX();
    //             lastResolutionY =
    //             info->getSupportedVideoModes()[i].getResolutionY();
    //         }
    //     }
    // }
    lastResolutionX = 0;
    lastResolutionY = 0;
    const openni::SensorInfo *depthinfo =
        device.getSensorInfo(openni::SENSOR_DEPTH);
    if (depthinfo)
    {
        for (int i = 0; i < depthinfo->getSupportedVideoModes().getSize(); i++)
        {
            // printf("Depth info video %d %dx%d FPS %d f %d\n", i,
            //        depthinfo->getSupportedVideoModes()[i].getResolutionX(),
            //        depthinfo->getSupportedVideoModes()[i].getResolutionY(),
            //        depthinfo->getSupportedVideoModes()[i].getFps(),
            //        depthinfo->getSupportedVideoModes()[i].getPixelFormat());
            if ((depthinfo->getSupportedVideoModes()[i].getResolutionX() !=
                 lastResolutionX) ||
                (depthinfo->getSupportedVideoModes()[i].getResolutionY() !=
                 lastResolutionY))
            {
                nResolutionDepth++;
                lastResolutionX =
                    depthinfo->getSupportedVideoModes()[i].getResolutionX();
                lastResolutionY =
                    depthinfo->getSupportedVideoModes()[i].getResolutionY();
            }
        }
    }
    rc = depth.create(device, openni::SENSOR_DEPTH);
    if (rc == openni::STATUS_OK)
    {
        rc = depth.start();
        if (rc != openni::STATUS_OK)
        {
            printf("SimpleViewer: Couldn't start depth stream:\n%s\n",
                   openni::OpenNI::getExtendedError());
            depth.destroy();
        }
    }
    else
    {
        printf("SimpleViewer: Couldn't find depth stream:\n%s\n",
               openni::OpenNI::getExtendedError());
    }

    rc = color.create(device, openni::SENSOR_COLOR);
    if (rc == openni::STATUS_OK)
    {
        openni::VideoMode vm;
        vm = color.getVideoMode();
        vm.setResolution(1280, 960);
        color.setVideoMode(vm);
        rc = color.start();
        if (rc != openni::STATUS_OK)
        {
            printf("SimpleViewer: Couldn't start color stream:\n%s\n",
                   openni::OpenNI::getExtendedError());
            color.destroy();
        }
    }
    else
    {
        printf("SimpleViewer: Couldn't find color stream:\n%s\n",
               openni::OpenNI::getExtendedError());
    }
    AXonLinkCamParam camParam;
    int dataSize = sizeof(AXonLinkCamParam);
    device.getProperty(AXONLINK_DEVICE_PROPERTY_GET_CAMERA_PARAMETERS,
                       &camParam, &dataSize);
    // for (int i = 0; i < nResolutionColor; i++)
    // {
    //     printf("astColorParam x =%d\n",
    //     camParam.astColorParam[i].ResolutionX); printf("astColorParam y
    //     =%d\n", camParam.astColorParam[i].ResolutionY); printf("astColorParam
    //     fx =%.5f\n", camParam.astColorParam[i].fx); printf("astColorParam fy
    //     =%.5f\n", camParam.astColorParam[i].fy); printf("astColorParam cx
    //     =%.5f\n", camParam.astColorParam[i].cx); printf("astColorParam cy
    //     =%.5f\n", camParam.astColorParam[i].cy); printf("astColorParam k1
    //     =%.5f\n", camParam.astColorParam[i].k1); printf("astColorParam k2
    //     =%.5f\n", camParam.astColorParam[i].k2); printf("astColorParam p1
    //     =%.5f\n", camParam.astColorParam[i].p1); printf("astColorParam p2
    //     =%.5f\n", camParam.astColorParam[i].p2); printf("astColorParam k3
    //     =%.5f\n", camParam.astColorParam[i].k3); printf("astColorParam k4
    //     =%.5f\n", camParam.astColorParam[i].k4); printf("astColorParam k5
    //     =%.5f\n", camParam.astColorParam[i].k5); printf("astColorParam k6
    //     =%.5f\n", camParam.astColorParam[i].k6);
    // }
    // for (int i = 0; i < nResolutionDepth; i++)
    // {
    //     printf("astDepthParam x =%d\n",
    //     camParam.astDepthParam[i].ResolutionX); printf("astDepthParam y
    //     =%d\n", camParam.astDepthParam[i].ResolutionY); printf("astDepthParam
    //     fx =%.5f\n", camParam.astDepthParam[i].fx); printf("astDepthParam fy
    //     =%.5f\n", camParam.astDepthParam[i].fy); printf("astDepthParam cx
    //     =%.5f\n", camParam.astDepthParam[i].cx); printf("astDepthParam cy
    //     =%.5f\n", camParam.astDepthParam[i].cy); printf("astDepthParam k1
    //     =%.5f\n", camParam.astDepthParam[i].k1); printf("astDepthParam k2
    //     =%.5f\n", camParam.astDepthParam[i].k2); printf("astDepthParam p1
    //     =%.5f\n", camParam.astDepthParam[i].p1); printf("astDepthParam p2
    //     =%.5f\n", camParam.astDepthParam[i].p2); printf("astDepthParam k3
    //     =%.5f\n", camParam.astDepthParam[i].k3); printf("astDepthParam k4
    //     =%.5f\n", camParam.astDepthParam[i].k4); printf("astDepthParam k5
    //     =%.5f\n", camParam.astDepthParam[i].k5); printf("astDepthParam k6
    //     =%.5f\n", camParam.astDepthParam[i].k6);
    // }
    // printf("R = %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f \n",
    // camParam.stExtParam.R_Param[0], camParam.stExtParam.R_Param[1],
    // camParam.stExtParam.R_Param[2], camParam.stExtParam.R_Param[3],
    // camParam.stExtParam.R_Param[4], camParam.stExtParam.R_Param[5],
    // camParam.stExtParam.R_Param[6], camParam.stExtParam.R_Param[7],
    // camParam.stExtParam.R_Param[8]); printf("T = %.5f %.5f %.5f \n",
    // camParam.stExtParam.T_Param[0], camParam.stExtParam.T_Param[1],
    // camParam.stExtParam.T_Param[2]);
    rc = ir.create(device, openni::SENSOR_IR);
    if (rc == openni::STATUS_OK)
    {
        rc = ir.start();
        if (rc != openni::STATUS_OK)
        {
            printf("SimpleViewer: Couldn't start color stream:\n%s\n",
                   openni::OpenNI::getExtendedError());
            ir.destroy();
        }
    }
    else
    {
        printf("SimpleViewer: Couldn't find color stream:\n%s\n",
               openni::OpenNI::getExtendedError());
    }
    AXonLinkGetExposureLevel value;
    int nSize = sizeof(value);
    ir.getProperty(AXONLINK_STREAM_PROPERTY_EXPOSURE_LEVEL, &value, &nSize);
    // printf("Get level:custId=%d,max=%d,current=%d\n", value.customID,
    // value.maxLevel, value.curLevel);

    if (!depth.isValid() || !color.isValid() || !ir.isValid())
    {
        printf("SimpleViewer: No valid streams. Exiting\n");
        openni::OpenNI::shutdown();
        return 2;
    }
    // rc = device.setDepthColorSyncEnabled(true);
    if (rc != openni::STATUS_OK)
    {
        printf("start sync failed1\n");
        openni::OpenNI::shutdown();
        return 4;
    }
    printf("[debug] Init axon depth camera done\n");
}

cAxonManager::cAxonManager()
{
    openni::Status rc = openni::STATUS_OK;
    openni::VideoStream color;
    int ret = InitDevice(rc, m_device, m_depthStream, color, m_irStream);
    Init();
}

void cAxonManager::Init()
{
    openni::VideoMode depthVideoMode;
    openni::VideoMode colorVideoMode;
    openni::VideoMode irVideoMode;

    if (m_depthStream.isValid())
    {
        depthVideoMode = m_depthStream.getVideoMode();

        int depthWidth = depthVideoMode.getResolutionX();
        int depthHeight = depthVideoMode.getResolutionY();
        // int colorWidth = colorVideoMode.getResolutionX();
        // int colorHeight = colorVideoMode.getResolutionY();

        m_width = depthWidth;
        m_height = depthHeight;
        // if (depthWidth == colorWidth &&
        //     depthHeight == colorHeight)
        // {
        // }
        // else
        // {
        //     /*     printf("Error - expect color and depth to be in same
        //     resolution: D: %dx%d, C: %dx%d\n",
        //         depthWidth, depthHeight,
        //         colorWidth, colorHeight);
        //     return openni::STATUS_ERROR;
        // 	*/
        //     m_width = colorWidth;
        //     m_height = colorHeight;
        // }
    }
    if (m_irStream.isValid())
    {
        irVideoMode = m_irStream.getVideoMode();
        int ir_width = irVideoMode.getResolutionX(),
            ir_height = irVideoMode.getResolutionY();
        if (ir_width != m_width || ir_height != m_height)
        {
            printf("[debug] ir shape %d %d != %d %d\n", ir_width, ir_height,
                   m_width, m_height);
        }
    }
    printf("[debug] width %d height %d\n", this->m_width, this->m_height);

    depth_mat.resize(m_height, m_width);
    ir_mat.resize(m_height, m_width);
    GetDepthImage();
    GetIrImage();
}

tMatrixXi cAxonManager::GetDepthImage()
{
    m_depthStream.readFrame(&m_depthFrame);
    const openni::DepthPixel *pDepthRow =
        (const openni::DepthPixel *)m_depthFrame.getData();
    // openni::RGB888Pixel *pTexRow = m_pTexMap + m_depthFrame.getCropOriginY()
    // * m_nTexMapX;
    int rowSize = m_depthFrame.getStrideInBytes() / sizeof(openni::DepthPixel);
    auto PixelFormat = m_depthFrame.getVideoMode().getPixelFormat();
    // printf("[debug] window size (%d, %d), cursor pos (%d, %d)\n",
    // m_depthFrame.getHeight(), m_depthFrame.getWidth(), m_curx, m_cury);
    int max = -1, min = 1e5;
    for (int y = 0; y < m_depthFrame.getHeight(); ++y)
    {
        const openni::DepthPixel *pDepth = pDepthRow;

        // openni::RGB888Pixel *pTex = pTexRow + m_depthFrame.getCropOriginX();

        // for (int x = 0; x < m_depthFrame.getWidth(); ++x, ++pDepth, ++pTex)
        for (int x = 0; x < m_depthFrame.getWidth(); ++x, ++pDepth)
        {
            uint16_t value = (*pDepth);
            depth_mat(y, m_depthFrame.getWidth() - x) = value;
            if (value > max)
                max = value;
            if (value < min)
                min = value;
            // if (*pDepth != 0 && *pDepth != 0XFFF)
            // {
            //     uint16_t value = (*pDepth);
            //     mat(y, x) = value;
            //     if (value > max)
            //         max = value;
            //     if (value < min)
            //         min = value;
            //     // printf("[debug] depth(%d, %d) = %.3f mm\n", x, y, value);
            //     // int nHistValue = m_pDepthHist[*pDepth];
            //     // if (m_curx == x && m_cury == y)
            //     // {
            //     // }
            //     // pTex->r = int(value);
            //     // pTex->g = int(value);
            //     // pTex->b = 0;
            // }
        }

        pDepthRow += rowSize;
        // pTexRow += m_nTexMapX;
    }
    // printf("[debug] max %d min %d (mm)\n", max, min);
    return this->depth_mat;
}

/**
 * \brief                   Get depth unit
 */
#include "OniEnums.h"

double cAxonManager::GetDepthUnit_mm()
{
    openni::PixelFormat format = m_depthFrame.getVideoMode().getPixelFormat();
    /*
    PIXEL_FORMAT_DEPTH_1_MM = 100,      //Depth data unit: 1mm
        PIXEL_FORMAT_DEPTH_100_UM = 101,    //Depth data unit: 100um (0.1mm)
        PIXEL_FORMAT_SHIFT_9_2 = 102,
        PIXEL_FORMAT_SHIFT_9_3 = 103,

        // Depth: AXon added
        PIXEL_FORMAT_DEPTH_1_3_MM = 110,    //Depth data unit: 1/3mm (0.333mm)
        PIXEL_FORMAT_DEPTH_1_2_MM = 111,	//Depth data unit: 1/2mm (0.5mm)
    */
    double res = 0;
    switch (format)
    {
    case openni::PixelFormat::PIXEL_FORMAT_DEPTH_1_MM:
        res = 1;
        break;

    case openni::PixelFormat::PIXEL_FORMAT_DEPTH_100_UM:
        res = 0.1;
        break;

    case openni::PixelFormat::PIXEL_FORMAT_DEPTH_1_3_MM:
        res = 0.333;
        break;

    case openni::PixelFormat::PIXEL_FORMAT_DEPTH_1_2_MM:
        res = 0.5;
        break;
    default:
        printf("[error] unrecognized depth init %d\n", format);
        exit(1);
    }
    return res;
}

tMatrixXi cAxonManager::GetIrImage()
{
    m_irStream.readFrame(&m_irFrame);
    if (m_irFrame.isValid())
    {
        // printf("ir %d %d\n", m_irFrame.getSensorType(),
        // m_irFrame.getFrameIndex());
        const OniGrayscale8Pixel *pDepthRow =
            (const OniGrayscale8Pixel *)m_irFrame.getData();
        // openni::RGB888Pixel *pTexRow = m_pTexMap + m_irFrame.getCropOriginY()
        // * m_nTexMapX;
        int rowSize = m_irFrame.getStrideInBytes() / sizeof(OniGrayscale8Pixel);
        for (int y = 0; y < m_irFrame.getHeight(); ++y)
        {
            const OniGrayscale8Pixel *pDepth = pDepthRow;
            // openni::RGB888Pixel *pTex = pTexRow + m_irFrame.getCropOriginX();

            // for (int x = 0; x < m_irFrame.getWidth(); ++x, ++pDepth, ++pTex)
            for (int x = 0; x < m_irFrame.getWidth(); ++x, ++pDepth)
            {
                // if (*pDepth != 0)
                {
                    uint8_t value = *pDepth;
                    ir_mat(y, x) = value;
                    // pTex->r = value;
                    // pTex->g = value;
                    // pTex->b = value;
                }
            }

            pDepthRow += rowSize;
            // pTexRow += m_nTexMapX;
        }
    }
    return ir_mat;
}

/**
 * \brief           Get intrinsic camera matrix (for depth camera)
 *
 *      mtx =
 *              \begin{bmatrix}
 *              f_x & 0 & c_x \\
 *              0 & f_y & c_y \\
 *              0 & 0 & 1 \\
 *              \end{bmatrix}
 */
tMatrix3d cAxonManager::GetDepthIntrinsicMtx() const
{
    AXonLinkCamParam camParam;
    int dataSize = sizeof(AXonLinkCamParam);
    m_device.getProperty(AXONLINK_DEVICE_PROPERTY_GET_CAMERA_PARAMETERS,
                         &camParam, &dataSize);
    tMatrix3d mtx = tMatrix3d::Identity();

    auto &param = camParam.astDepthParam[0];
    mtx(0, 0) = param.fx;
    mtx(1, 1) = param.fy;
    mtx(0, 2) = param.cx;
    mtx(1, 2) = param.cy;
    return mtx;
    // for (int i = 0; i < nResolutionDepth; i++)
    // {
    //     printf("astDepthParam x =%d\n",
    //     camParam.astDepthParam[i].ResolutionX); printf("astDepthParam y
    //     =%d\n", camParam.astDepthParam[i].ResolutionY); printf("astDepthParam
    //     fx =%.5f\n", camParam.astDepthParam[i].fx); printf("astDepthParam fy
    //     =%.5f\n", camParam.astDepthParam[i].fy); printf("astDepthParam cx
    //     =%.5f\n", camParam.astDepthParam[i].cx); printf("astDepthParam cy
    //     =%.5f\n", camParam.astDepthParam[i].cy); printf("astDepthParam k1
    //     =%.5f\n", camParam.astDepthParam[i].k1); printf("astDepthParam k2
    //     =%.5f\n", camParam.astDepthParam[i].k2); printf("astDepthParam p1
    //     =%.5f\n", camParam.astDepthParam[i].p1); printf("astDepthParam p2
    //     =%.5f\n", camParam.astDepthParam[i].p2); printf("astDepthParam k3
    //     =%.5f\n", camParam.astDepthParam[i].k3); printf("astDepthParam k4
    //     =%.5f\n", camParam.astDepthParam[i].k4); printf("astDepthParam k5
    //     =%.5f\n", camParam.astDepthParam[i].k5); printf("astDepthParam k6
    //     =%.5f\n", camParam.astDepthParam[i].k6);
    // }
}

/**
 * \brief               Get the intrinsic distortion coeffs (depth camera)
 *
 * dist_coef = k_1, k_2, p_1, p_2, k_3, k_4, k_5, k_6
 * size = 8
 * which has the same order and meaning with opencv2,
 * please check
 * https://docs.opencv.org/2.4.13.7/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#calibratecamera
 */
tVectorXd cAxonManager::GetDepthIntrinsicDistCoef() const
{
    int size = 8;
    tVectorXd dist_coef = tVectorXd::Zero(8);
    AXonLinkCamParam camParam;
    int dataSize = sizeof(AXonLinkCamParam);
    m_device.getProperty(AXONLINK_DEVICE_PROPERTY_GET_CAMERA_PARAMETERS,
                         &camParam, &dataSize);

    auto &param = camParam.astDepthParam[0];
    dist_coef[0] = param.k1;
    dist_coef[1] = param.k2;
    dist_coef[2] = param.p1;
    dist_coef[3] = param.p2;
    dist_coef[4] = param.k3;
    dist_coef[5] = param.k4;
    dist_coef[6] = param.k5;
    dist_coef[7] = param.k6;
    return dist_coef;
}