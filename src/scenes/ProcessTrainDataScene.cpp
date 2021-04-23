#include "ProcessTrainDataScene.h"
#include "utils/JsonUtil.h"
#include <iostream>
cProcessTrainDataScene::cProcessTrainDataScene()
{
}
cProcessTrainDataScene::~cProcessTrainDataScene()
{
}
void cProcessTrainDataScene::Init(const std::string &conf_path)
{
    Json::Value root;
    cJsonUtil::LoadJson(conf_path, root);
    InitCameraInfo(root);
}
void cProcessTrainDataScene::Update(double dt)
{
}
void cProcessTrainDataScene::UpdateRenderingResource()
{
}
void cProcessTrainDataScene::Reset()
{
}
void cProcessTrainDataScene::CursorMove(cDrawScene *draw_scene, int xpos, int ypos)
{
}
void cProcessTrainDataScene::MouseButton(cDrawScene *draw_scene, int button, int action,
                                         int mods)
{
}
void cProcessTrainDataScene::Key(int key, int scancode, int action, int mods)
{
}

/**
 * \brief           Init capture camera info
*/
void cProcessTrainDataScene::InitCameraInfo(const Json::Value &conf)
{
    mCameraCenter = cJsonUtil::ReadVectorJson(
                        cJsonUtil::ParseAsValue(cProcessTrainDataScene::CAMERA_CENTER_KEY, conf))
                        .segment(0, 4);
    mCameraUp = cJsonUtil::ReadVectorJson(
                    cJsonUtil::ParseAsValue(cProcessTrainDataScene::CAMERA_UP_KEY, conf))
                    .segment(0, 4);

    // std::cout << "center = " << mCameraCenter.transpose() << std::endl;
    // std::cout << "up = " << mCameraUp.transpose() << std::endl;
    Json::Value pos_lst = cJsonUtil::ParseAsValue(cProcessTrainDataScene::CAMERA_POS_KEY, conf);
    SIM_ASSERT(pos_lst.size() >= 1);
    mCameraPos.clear();
    for (int i = 0; i < pos_lst.size(); i++)
    {
        tVector tmp = cJsonUtil::ReadVectorJson(pos_lst[i]).segment(0, 4);
        // std::cout << "pos " << i << " = " << tmp.transpose() << std::endl;
        mCameraPos.push_back(tmp);
    }
    // exit(0);
}

void cProcessTrainDataScene::UpdateSubstep()
{
}