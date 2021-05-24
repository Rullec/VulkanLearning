#pragma once
#include "SimScene.h"
#include "utils/DefUtil.h"

SIM_DECLARE_CLASS_AND_PTR(CameraBase);
class cProcessTrainDataScene : public cSimScene
{
public:
    cProcessTrainDataScene();
    virtual ~cProcessTrainDataScene();
    void InitExport(std::string conf_path);
    virtual void Init(const std::string &conf_path) override;
    virtual void Update(double dt) override;
    virtual void UpdateRenderingResource() override;
    virtual void Reset() override;

protected:
    inline static const std::string CAMERA_POS_KEY = "capture_camera_pos",
                                    CAMERA_CENTER_KEY = "capture_camera_center",
                                    CAMERA_UP_KEY = "capture_camera_up",
                                    CAMERA_FOV_KEY = "capture_camera_fov",
                                    GEOMETRY_INFO_KEY = "geometry_info",
                                    RAW_DATA_DIR_KEY = "raw_data_dir",
                                    GEN_DATA_DIR_KEY = "gen_data_dir",
                                    DEPTH_IMAGE_WIDTH_KEY = "window_width",
                                    DEPTH_IMAGE_HEIGHT_KEY = "window_height",
                                    SIMULATION_CONF_KEY = "simulation_conf",
                                    ENABLE_CLOTH_GEOMETRY_KEY =
                                        "enable_cloth_geometry";
    std::string mGeometryInfoPath; // the geometry info
    std::string mRawDataDir;       // raw simulation data dir
    std::string mGenDataDir;       // gen new data dir
    bool mEnableClothGeometry;     // add cloth geometry into the raycast scene
    int mWidth, mHeight;
    tVector mCameraPos, mCameraCenter,
        mCameraUp; // camera position, center point and up direction
    float mCameraFov;
    CameraBasePtr mCamera; // camera pos
    void InitCameraInfo(const Json::Value &conf);
    virtual void UpdateSubstep() override final;
    bool LoadRawData(std::string path, tVectorXd &feature_vec);
    // tMatrixXd CalcDepthImageLegacy(const CameraBasePtr camera);
    void CalcDepthMap(

        const std::string raw_data_path, const std::string &save_png_path,
        const std::string &json_path, CameraBasePtr camera);
    // void CalcDepthMapMultiViews(const std::string raw_data_path, const
    // std::vector<std::string> &save_png_path_array, const
    // std::vector<std::string> &json_path_array, const
    // std::vector<CameraBasePtr> &camera_array);
    void CalcDepthMapMultiViews(const std::string &surface_geo_path,
                                const std::string &basename,
                                const std::vector<CameraBasePtr> &camera_array,
                                int num_of_rotation_view);
    void MultiViewTest();
    void InitCameraViews();
    virtual void InitRaycaster() override;
    void CalcDepthMapLoop();
    void CalcDepthMapNoCloth();
};