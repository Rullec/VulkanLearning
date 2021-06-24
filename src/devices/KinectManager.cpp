#include "KinectManager.h"
#include <iostream>
#include <k4a/k4a.h>
#define SIM_ASSERT assert
cKinectManager::cKinectManager(std::string mode_str)
{
    mDepthMode = BuildModeFromString(mode_str);
    Init();
}

#include <windows.h>
k4a_capture_t cKinectManager::GetCapture() const
{
    bool get_capture = false;

    k4a_capture_t capture = NULL;
    const int32_t TIMEOUT_IN_MS = 1000;

    while (get_capture == false)
    {
        // k4a_image_t image;

        // Get a depth frame
        switch (k4a_device_get_capture(mDevice, &capture, TIMEOUT_IN_MS))
        {
        case K4A_WAIT_RESULT_SUCCEEDED:
            get_capture = true;
            // std::cout << "case succ\n";
            break;
        case K4A_WAIT_RESULT_TIMEOUT:
            // std::cout << "case timeout\n";
            printf("Timed out waiting for a capture\n");
            continue;
            break;
        case K4A_WAIT_RESULT_FAILED:
            // std::cout << "case failed\n";
            SIM_ASSERT(false && "get capture failed");
            exit(0);
            break;
        }
    }
    // std::cout << "get capture done \n";
    // exit(0);
    return capture;
}

tMatrixXi convert_k4a_depth_image_to_eigen_image(k4a_image_t image)
{
    int height = 0, width = 0, stride_bytes = 0;
    if (image != NULL)
    {
        height = k4a_image_get_height_pixels(image);
        width = k4a_image_get_width_pixels(image);
        stride_bytes = k4a_image_get_stride_bytes(image);
    }
    else
    {
        printf(" | Depth16 None\n");
        SIM_ASSERT(false);
    }
    auto image_format = k4a_image_get_format(image);

    assert(image_format == K4A_IMAGE_FORMAT_DEPTH16);
    // printf("[debug] get depth image height %d, width %d, stride bytes %d\n",
    // height, width, stride_bytes); exit(0);
    uint8_t *buffer_raw = k4a_image_get_buffer(image);
    tMatrixXi depth_mat = tMatrixXi::Zero(height, width);
    if (image_format == K4A_IMAGE_FORMAT_DEPTH16)
    {
        // std::cout << "* Each pixel of DEPTH16 data is two bytes of little
        // endian unsigned depth data. The unit of the data is in millimeters
        // from the origin of the camera\n";
        uint16_t *depth_buffer = (uint16_t *)buffer_raw;
        // row major

        for (int row_id = 0; row_id < height; row_id++)
        {
            for (int col_id = 0; col_id < width; col_id++)
            {
                depth_mat(row_id, col_id) =
                    depth_buffer[row_id * width + col_id];
            }
            // depth_buffer += stride_bytes / 2;
        }
    }
    else
    {
        assert(false && "supported depth image type");
    }
    return depth_mat;
}
/**
 * \brief           Begin to get the depth image
 */
tMatrixXi cKinectManager::GetDepthImage()
{
    // get captured
    auto capture = GetCapture();
    k4a_image_t image = k4a_capture_get_depth_image(capture);
    tMatrixXi depth_mat = convert_k4a_depth_image_to_eigen_image(image);
    // tMatrixXi
    // release the image
    k4a_image_release(image);
    k4a_capture_release(capture);
    return depth_mat;
}

double cKinectManager::GetDepthUnit_mm() { return 1; }
#include "utils/TimeUtil.hpp"
tMatrixXi cKinectManager::GetIrImage()
{
    // cTimeUtil::Begin("get_ir_image");

    auto capture = GetCapture();

    k4a_image_t image = k4a_capture_get_ir_image(capture);
    int height = 0, width = 0, stride_bytes = 0;
    if (image != NULL)
    {
        height = k4a_image_get_height_pixels(image);
        width = k4a_image_get_width_pixels(image);
        stride_bytes = k4a_image_get_stride_bytes(image);
    }
    else
    {
        printf(" | ir None\n");
        SIM_ASSERT(false);
    }
    auto image_format = k4a_image_get_format(image);

    assert(image_format == K4A_IMAGE_FORMAT_IR16);
    // printf("[debug] get depth image height %d, width %d, stride bytes %d\n",
    // height, width, stride_bytes); exit(0);
    uint8_t *buffer_raw = k4a_image_get_buffer(image);
    tMatrixXi depth_mat = tMatrixXi::Zero(height, width);
    if (image_format == K4A_IMAGE_FORMAT_IR16)
    {
        // std::cout << "* Each pixel of IR16 data is two bytes of little endian
        // unsigned depth data. The unit of the data is in millimeters from the
        // origin of the camera\n";
        uint16_t *depth_buffer = (uint16_t *)buffer_raw;
        // row major

        for (int row_id = 0; row_id < height; row_id++)
        {
            for (int col_id = 0; col_id < width; col_id++)
            {
                depth_mat(row_id, col_id) =
                    depth_buffer[row_id * width + col_id];
            }
            // depth_buffer += stride_bytes / 2;
        }
    }
    else
    {
        assert(false && "supported depth image type");
    }
    // tMatrixXi
    // release the image
    k4a_image_release(image);
    k4a_capture_release(capture);
    // cTimeUtil::End("get_ir_image");
    return depth_mat;
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

tMatrix3d cKinectManager::GetDepthIntrinsicMtx_sdk() const
{
    auto depth_calib = GetDepthCalibration();
    tMatrix3d mat = tMatrix3d::Identity();

    mat(0, 0) = depth_calib.intrinsics.parameters.param.fx;
    mat(1, 1) = depth_calib.intrinsics.parameters.param.fy;
    mat(0, 2) = depth_calib.intrinsics.parameters.param.cx;
    mat(1, 2) = depth_calib.intrinsics.parameters.param.cy;
    return mat;
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
tVectorXd cKinectManager::GetDepthIntrinsicDistCoef_sdk() const
{
    auto depth_calib = GetDepthCalibration();
    tVectorXd res = tVectorXd::Zero(8);
    res[0] = depth_calib.intrinsics.parameters.param.k1;
    res[1] = depth_calib.intrinsics.parameters.param.k2;
    res[2] = depth_calib.intrinsics.parameters.param.p1;
    res[3] = depth_calib.intrinsics.parameters.param.p2;
    res[4] = depth_calib.intrinsics.parameters.param.k3;
    res[5] = depth_calib.intrinsics.parameters.param.k4;
    res[6] = depth_calib.intrinsics.parameters.param.k5;
    res[7] = depth_calib.intrinsics.parameters.param.k6;
    return res;
}

/**
 * \brief                   Init kinect device
 */
void cKinectManager::Init()
{
    // counting the devices
    uint32_t device_count = k4a_device_get_installed_count();
    // std::cout << "device count = " << device_count << std::endl;
    SIM_ASSERT(device_count == 1);

    // open the device
    auto res = k4a_device_open(K4A_DEVICE_DEFAULT, &mDevice);
    // std::cout << "k4a device open res = " << res << std::endl;
    SIM_ASSERT(res == K4A_RESULT_SUCCEEDED);

    // set up the configuration
    mConfig = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    mConfig.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
    // mConfig.color_format = K4A_IMAGE_FORMAT_COLOR_MJPG;
    mConfig.color_resolution = K4A_COLOR_RESOLUTION_720P;
    // mConfig.color_resolution = K4A_COLOR_RESOLUTION_1080P;
    mConfig.depth_mode = mDepthMode;
    // mConfig.depth_mode = K4A_DEPTH_MODE_WFOV_UNBINNED;
    mConfig.camera_fps = K4A_FRAMES_PER_SECOND_30;
    mConfig.synchronized_images_only = true;
    // mConfig.camera_fps = K4A_FRAMES_PER_SECOND_15;

    // start the camera
    if (K4A_RESULT_SUCCEEDED != k4a_device_start_cameras(mDevice, &mConfig))
    {
        printf("[debug] start the camera failed, exit\n");
        exit(1);
    }
}

cKinectManager::~cKinectManager() { k4a_device_close(this->mDevice); }

k4a_calibration_camera_t cKinectManager::GetDepthCalibration() const
{
    k4a_calibration_t calibration;
    // get calibration
    if (K4A_RESULT_SUCCEEDED !=
        k4a_device_get_calibration(this->mDevice, this->mConfig.depth_mode,
                                   mConfig.color_resolution, &calibration))
    {
        std::cout << "Failed to get calibration" << std::endl;
        exit(-1);
    }
    return calibration.depth_camera_calibration;
}

k4a_calibration_camera_t cKinectManager::GetColorCalibration() const
{
    k4a_calibration_t calibration;
    // get calibration
    if (K4A_RESULT_SUCCEEDED !=
        k4a_device_get_calibration(this->mDevice, this->mConfig.depth_mode,
                                   mConfig.color_resolution, &calibration))
    {
        std::cout << "Failed to get calibration" << std::endl;
        exit(-1);
    }
    return calibration.color_camera_calibration;
}
/**
 * \brief               Get intrinsics self
 */
tMatrix3d cKinectManager::GetDepthIntrinsicMtx_self() const
{
    /*
         k1 k2 p1 p2 k3
dist : -0.329146, 0.166924, 0.00371129, 0.00137014, -0.0609417
        fx fy cx cy skew
cam matrix: , , , , -0.00683854
    */
    tMatrix3d mat = tMatrix3d::Identity();
    mat(0, 0) = 498.417;
    mat(1, 1) = 499.322;
    mat(0, 2) = 502.074;
    mat(1, 2) = 491.664;
    return mat;
}

/**
 * \brief               Get intrinsics self
 */
tVectorXd cKinectManager::GetDepthIntrinsicDistCoef_self() const
{
    int size = 8;
    tVectorXd coef = tVectorXd::Zero(size);
    /*
    coming from shining 3d
    -0.329146, 0.166924, 0.00371129, 0.00137014, -0.0609417
    */
    double k1 = -0.329146, k2 = 0.166924, p1 = 0.00371129, p2 = 0.00137014,
           k3 = -0.0609417;

    // double k1 = -0.35272165, k2 = 0.21301618, p1 = 0.00066491, p2 =
    // -0.00061842,
    //        k3 = -0.10257608;
    coef[0] = k1;
    coef[1] = k2;
    coef[2] = p1;
    coef[3] = p2;
    coef[4] = k3;
    return coef;
}
#include "utils/DefUtil.h"

/**
 * \brief       get depth mode (string mode)
 */
std::string cKinectManager::GetDepthMode() const
{
    return BuildStringFromDepthMode(mDepthMode);
}

/**
 * \brief           set depth mode
 */
void cKinectManager::SetDepthMode(std::string mode_str)
{
    k4a_depth_mode_t new_mode = BuildModeFromString(mode_str);

    if (new_mode != this->mDepthMode)
    {
        this->CloseDevice();
        mDepthMode = new_mode;
        Init();
    }
}

std::string cKinectManager::BuildStringFromDepthMode(k4a_depth_mode_t mode)
{
    std::string mode_str;
    switch (mode)
    {
    case K4A_DEPTH_MODE_NFOV_UNBINNED:
        mode_str = "nfov_unbinned";
        break;
    case K4A_DEPTH_MODE_PASSIVE_IR:
        mode_str = "passive_ir";
        break;
    default:
        std::cout << "build string failed for mode " << mode << std::endl;
        exit(0);
        break;
    }
    return mode_str;
}
k4a_depth_mode_t cKinectManager::BuildModeFromString(std::string mode_str)
{
    k4a_depth_mode_t mode;
    if (mode_str == "nfov_unbinned")
    {
        mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
    }
    else if (mode_str == "passive_ir")
    {
        mode = K4A_DEPTH_MODE_PASSIVE_IR;
    }
    else
    {
        std::cout << "unsupported mode_str " << mode_str << std::endl;
        exit(1);
    }
    return mode;
}

void cKinectManager::CloseDevice() { k4a_device_close(mDevice); }

/**
 * \brief           get rgb image from the device
 */
std::vector<tMatrixXi> cKinectManager::GetColorImage() const
{
    auto capture = GetCapture();
    auto color_image = k4a_capture_get_color_image(capture);
    if (color_image == 0)
    {
        printf("Failed to get color image from capture\n");
        exit(1);
    }

    // convert this color image into the eigen format
    int height = k4a_image_get_height_pixels(color_image);
    int width = k4a_image_get_width_pixels(color_image);
    uint8_t *image_data = (uint8_t *)(void *)k4a_image_get_buffer(color_image);
    std::vector<tMatrixXi> bgra_images(4, tMatrixXi::Zero(height, width));
    for (int row = 0; row < height; row++)
        for (int col = 0; col < width; col++)
        {
            int offset = (row * width + col) * 4;
            for (int j = 0; j < 4; j++)
            {
                bgra_images[j](row, col) = image_data[offset + j];
            }

            // ;
            // image_data[offset + 1];
            // image_data[offset + 2];
            // image_data[offset + 3];
        }
    return bgra_images;
    // cv::Mat color_frame =
    //     cv::Mat(height, width, CV_8UC4, image_data, cv::Mat::AUTO_STEP);
}

static tMatrixXi
point_cloud_depth_to_color(k4a_transformation_t transformation_handle,
                           const k4a_image_t depth_image,
                           const k4a_image_t color_image)
{
    // transform color image into depth camera geometry
    int color_image_width_pixels = k4a_image_get_width_pixels(color_image);
    int color_image_height_pixels = k4a_image_get_height_pixels(color_image);
    k4a_image_t transformed_depth_image = NULL;
    if (K4A_RESULT_SUCCEEDED !=
        k4a_image_create(K4A_IMAGE_FORMAT_DEPTH16, color_image_width_pixels,
                         color_image_height_pixels,
                         color_image_width_pixels * (int)sizeof(uint16_t),
                         &transformed_depth_image))
    {
        printf("Failed to create transformed depth image\n");
        exit(1);
    }

    k4a_image_t point_cloud_image = NULL;
    if (K4A_RESULT_SUCCEEDED !=
        k4a_image_create(K4A_IMAGE_FORMAT_CUSTOM, color_image_width_pixels,
                         color_image_height_pixels,
                         color_image_width_pixels * 3 * (int)sizeof(int16_t),
                         &point_cloud_image))
    {
        printf("Failed to create point cloud image\n");
        exit(1);
    }

    if (K4A_RESULT_SUCCEEDED !=
        k4a_transformation_depth_image_to_color_camera(
            transformation_handle, depth_image, transformed_depth_image))
    {
        printf("Failed to compute transformed depth image\n");
        exit(1);
    }

    if (K4A_RESULT_SUCCEEDED !=
        k4a_transformation_depth_image_to_point_cloud(
            transformation_handle, transformed_depth_image,
            K4A_CALIBRATION_TYPE_COLOR, point_cloud_image))
    {
        printf("Failed to compute point cloud\n");
        exit(1);
    }

    tMatrixXi depth_image_eigen =
        convert_k4a_depth_image_to_eigen_image(transformed_depth_image);

    k4a_image_release(transformed_depth_image);
    k4a_image_release(point_cloud_image);

    return depth_image_eigen;
}

/**
 * \brief           get depth image which is aligned with the color image
 */
tMatrixXi cKinectManager::GetDepthToColorImage() const
{
    k4a_calibration_t calibration;
    if (K4A_RESULT_SUCCEEDED !=
        k4a_device_get_calibration(mDevice, mConfig.depth_mode,
                                   mConfig.color_resolution, &calibration))
    {
        printf("Failed to get calibration\n");
        exit(1);
    }

    k4a_transformation_t transformation =
        k4a_transformation_create(&calibration);

    // Get a capture
    k4a_capture_t capture = GetCapture();

    // Get a depth image
    auto depth_image = k4a_capture_get_depth_image(capture);
    if (depth_image == 0)
    {
        printf("Failed to get depth image from capture\n");
        exit(1);
    }

    // Get a color image
    auto color_image = k4a_capture_get_color_image(capture);
    if (color_image == 0)
    {
        printf("Failed to get color image from capture\n");
        exit(1);
    }

    // Compute color point cloud by warping depth image into color camera
    // geometry
    tMatrixXi depth_image_eigen =
        point_cloud_depth_to_color(transformation, depth_image, color_image);
    return depth_image_eigen;
}

/**
 * \brief           get the intrinsics for color camera
 */
tMatrix3d cKinectManager::GetColorIntrinsicMtx_sdk() const
{
    auto color_calib = GetColorCalibration();
    tMatrix3d mat = tMatrix3d::Identity();

    mat(0, 0) = color_calib.intrinsics.parameters.param.fx;
    mat(1, 1) = color_calib.intrinsics.parameters.param.fy;
    mat(0, 2) = color_calib.intrinsics.parameters.param.cx;
    mat(1, 2) = color_calib.intrinsics.parameters.param.cy;
    return mat;
}

/**
 * \brief           get the intrinsics for color camera
 */
tVectorXd cKinectManager::GetColorIntrinsicDistCoef_sdk() const
{
    auto color_calib = GetColorCalibration();
    tVectorXd res = tVectorXd::Zero(8);
    res[0] = color_calib.intrinsics.parameters.param.k1;
    res[1] = color_calib.intrinsics.parameters.param.k2;
    res[2] = color_calib.intrinsics.parameters.param.p1;
    res[3] = color_calib.intrinsics.parameters.param.p2;
    res[4] = color_calib.intrinsics.parameters.param.k3;
    res[5] = color_calib.intrinsics.parameters.param.k4;
    res[6] = color_calib.intrinsics.parameters.param.k5;
    res[7] = color_calib.intrinsics.parameters.param.k6;
    return res;
}