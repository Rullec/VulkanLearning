
#include "BaraffScene.h"

cBaraffScene::cBaraffScene() {}

void cBaraffScene::Init(const std::string &conf_path)
{
    cSimScene::Init(conf_path);
    InitRaycaster();
}

cBaraffScene::~cBaraffScene() {}

void cBaraffScene::UpdateSubstep() {}