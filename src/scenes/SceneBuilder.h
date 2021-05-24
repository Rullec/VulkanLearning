
#pragma once
#include "scenes/DrawScene.h"
#include "scenes/Scene.h"
#include "scenes/SimScene.h"
#include "utils/DefUtil.h"
#include <memory>
#include <string>
class cSceneBuilder
{
public:
    static std::shared_ptr<cScene> BuildScene(const std::string type,
                                              bool enable_draw = true);
    static std::shared_ptr<cSimScene>
    BuildSimScene(const std::string config_file);
};