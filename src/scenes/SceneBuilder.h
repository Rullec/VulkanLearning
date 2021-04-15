
#pragma once
#include "scenes/DrawScene.h"
#include <memory>
#include <string>

class cSceneBuilder
{
public:
    static std::shared_ptr<cDrawScene> BuildScene(const std::string type, bool enable_draw = true);
    static std::shared_ptr<cSimScene> BuildSimScene(const std::string config_file);
};