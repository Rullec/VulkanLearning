
#pragma once
#include <memory>
#include <string>
#include "utils/DefUtil.h"
#include "scenes/Scene.h"
#include "scenes/DrawScene.h"
#include "scenes/SimScene.h"
class cSceneBuilder
{
public:
    static std::shared_ptr<cScene> BuildScene(const std::string type, bool enable_draw = true);
    static std::shared_ptr<cSimScene> BuildSimScene(const std::string config_file);
};