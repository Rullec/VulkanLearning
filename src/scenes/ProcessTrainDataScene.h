#ifdef _WIN32
#pragma once
#include "SimScene.h"
#include "utils/DefUtil.h"

SIM_DECLARE_CLASS_AND_PTR(CameraBase);
class cProcessTrainDataScene : public cSimScene
{
public:
    cProcessTrainDataScene();
    virtual ~cProcessTrainDataScene();
    virtual void Init(const std::string &conf_path) override;
    virtual void Update(double dt) override;
    virtual void UpdateRenderingResource() override;
    virtual void Reset() override;
    std::pair<int, int> GetDepthImageShape() const;
    tVectorXf CalcEmptyDepthImage(const tVector &cam_pos,
                                  const tVector &cam_focus, float fov);
    bool GetEnableOnlyExportingCuttedWindow() const;
    Eigen::Matrix2i GetCuttedWindow() const;
    tVector2i GetResolution() const;

protected:
    enum eImageType
    {
        EXR_TYPE,
        PNG_TYPE,
        NUM_OF_TYPES
    };
    inline static const std::string gImageSuffix[eImageType::NUM_OF_TYPES] = {
        ".exr",
        ".png",
    };
    static eImageType GetImageType(const std::string name);
    static std::string GetImageSuffix(eImageType type);

    inline static const std::string
        CAMERA_POS_KEY = "capture_camera_pos",
        CAMERA_CENTER_KEY = "capture_camera_center",
        CAMERA_UP_KEY = "capture_camera_up",
        CAMERA_FOV_KEY = "capture_camera_fov",
        GEOMETRY_INFO_KEY = "geometry_info", RAW_DATA_DIR_KEY = "raw_data_dir",
        GEN_DATA_DIR_KEY = "gen_data_dir",
        DEPTH_IMAGE_WIDTH_KEY = "window_width",
        DEPTH_IMAGE_HEIGHT_KEY = "window_height",
        SIMULATION_CONF_KEY = "simulation_conf",
        ENABLE_CLOTH_GEOMETRY_KEY = "enable_cloth_geometry",
        ENABLE_CAMERA_NOISE_KEY = "enable_camera_noise",
        CAMERA_TRANSLATION_NOISE_KEY = "camera_translation_noise",
        CAMERA_ORIENTATION_NOISE_KEY = "camera_orientation_noise",
        CAMERA_NOISE_SAMPLES_KEY = "camera_noise_samples",
        EXPORT_IMAGE_FORMAT_KEY = "export_image_format",
        CASTING_WIDTH_RANGE_KEY = "casting_width_range",
        CASTING_HEIGHT_RANGE_KEY = "casting_height_range",
        NUM_OF_CLOTH_VIEWS_KEY = "num_of_cloth_views",
        NUM_OF_INIT_ROTATION_ANGLE_KEY = "num_of_init_rotation_angle",
        UPSAMPLING_KEY = "upsampling";

    struct
    {
        int mNumOfInitRotationAngles; // step1: init cloth rotation angles
        bool mEnableCameraNoise; // step2: enable camera noise on orientation &
                                 // position
        double mCameraTranslationNoise,
            mCameraOrientationNoise;  // the amptitude of camera translation &
                                      // orientation
        int mCameraNoiseSamples;      // how many samples do we have?
        int mNumOfClothRotationViews; // step3: num of views for rotating the
                                      // cloth
    } mPreprocessInfo;

    Eigen::Matrix2i mCastingRange; // casting window screen coordinates
    eImageType mExportImageType;
    double mImageUpsampling;
    std::string mGeometryInfoPath; // the geometry info
    std::string mRawDataDir;       // raw simulation data dir
    std::string mGenDataDir;       // gen new data dir
    bool mEnableClothGeometry;     // add cloth geometry into the raycast scene
    // bool mEnableOnlyExportingCuttedWindow; // only export the cutted window to
    //                                        // png file

    int mWidth, mHeight;
    tVector mCameraPos, mCameraCenter,
        mCameraUp; // camera position, center point and up direction
    float mCameraFov;
    std::vector<CameraBasePtr> mCameraLst; // camera pointer

    // apply noise on the camera position and orientation
    void InitPreprocessInfo(const Json::Value &conf);
    // virtual void UpdateSubstep() override final;
    bool LoadRawData(std::string path, tVectorXd &feature_vec);
    // tMatrixXd CalcDepthImageLegacy(const CameraBasePtr camera);
    void CalcDepthMap(

        const std::string raw_data_path, const std::string &save_png_path,
        const std::string &json_path, CameraBasePtr camera);
    void CalcDepthMapMultiViews(const std::string &output_dir,
                                const std::vector<CameraBasePtr> &camera_array,
                                int num_of_rotation_view);
    void MultiViewTest();
    void GenerateCameraViews();
    void InitObstacle(const Json::Value &conf);
    virtual void InitRaycaster(const Json::Value & conf) override;
    void CalcDepthMapLoop();
    void CalcDepthMapNoCloth();
    static void ValidateOutputDir(std::string dir);
};
#endif