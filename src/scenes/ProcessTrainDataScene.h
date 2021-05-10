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

protected:
    inline static const std::string CAMERA_POS_KEY = "capture_camera_pos",
                                    CAMERA_CENTER_KEY = "capture_camera_center",
                                    CAMERA_UP_KEY = "capture_camera_up",
                                    GEOMETRY_INFO_KEY = "geometry_info",
                                    RAW_DATA_DIR_KEY = "raw_data_dir",
                                    GEN_DATA_DIR_KEY = "gen_data_dir",
                                    DEPTH_IMAGE_WIDTH_KEY = "window_width",
                                    DEPTH_IMAGE_HEIGHT_KEY = "window_height";
    std::string mGeometryInfoPath; // the geometry info
    std::string mRawDataDir;       // raw simulation data dir
    std::string mGenDataDir;       // gen new data dir
    int mWidth, mHeight;
    tVector mCameraCenter, mCameraUp;        // camera center point and up direction
    tEigenArr<tVector> mCameraPos;           // a list of camera position
    std::vector<CameraBasePtr> mCameraViews; // camera pos
    void InitCameraInfo(const Json::Value &conf);
    virtual void UpdateSubstep() override final;
    bool LoadRawData(std::string path, tVectorXd &feature_vec);
    tMatrixXd CalcDepthImageLegacy(const CameraBasePtr camera);
    void CalcDepthMap(const std::string raw_data_path, const std::string &save_png_path, const std::string &json_path, CameraBasePtr camera);
    void CalcDepthMapMultiViews(const std::string raw_data_path, const std::vector<std::string> &save_png_path_array, const std::vector<std::string> &json_path_array, const std::vector<CameraBasePtr> &camera_array);
    void MultiViewTest();
    void InitCameraViews();
};