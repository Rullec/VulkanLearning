#pragma once
#include "SimScene.h"

class cProcessTrainDataScene : public cSimScene
{
public:
    cProcessTrainDataScene();
    virtual ~cProcessTrainDataScene();
    virtual void Init(const std::string &conf_path) override;
    virtual void Update(double dt) override;
    virtual void UpdateRenderingResource() override;
    virtual void Reset() override;
    virtual void CursorMove(cDrawScene *draw_scene, int xpos, int ypos) override;
    virtual void MouseButton(cDrawScene *draw_scene, int button, int action,
                             int mods) override;
    virtual void Key(int key, int scancode, int action, int mods) override;

protected:
    inline static const std::string CAMERA_POS_KEY = "capture_camera_pos",
                                    CAMERA_CENTER_KEY = "capture_camera_center",
                                    CAMERA_UP_KEY = "capture_camera_up";
    tVector mCameraCenter, mCameraUp; // camera center point and up direction
    tEigenArr<tVector> mCameraPos;    // a list of camera position
    void InitCameraInfo(const Json::Value &conf);
    virtual void UpdateSubstep() override final;
};