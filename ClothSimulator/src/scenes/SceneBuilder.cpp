#include "SceneBuilder.h"
#include "SimScene.h"
#include "MassSpringScene.h"
#include "utils/LogUtil.h"
#include "utils/JsonUtil.h"
#include "TrimeshScene.h"

std::shared_ptr<cDrawScene> cSceneBuilder::BuildScene(const std::string type, bool enable_draw /*= true*/)
{
    if (enable_draw == true)
    {
        return std::make_shared<cDrawScene>();
    }
    else
    {
        SIM_ASSERT(false);
    }
}
std::shared_ptr<cSimScene> cSceneBuilder::BuildSimScene(const std::string config_file)
{
    Json::Value root;
    cJsonUtil::LoadJson(config_file, root);
    std::string type = cJsonUtil::ParseAsString("integration_scheme", root);
    eIntegrationScheme scheme = cSimScene::BuildIntegrationScheme(type);
    std::shared_ptr<cSimScene> scene = nullptr;
    switch (scheme)
    {
    case eIntegrationScheme::MS_IMPLICIT:
    case eIntegrationScheme::MS_OPT_IMPLICIT:
    case eIntegrationScheme::MS_SEMI_IMPLICIT:
        scene = std::make_shared<cMSScene>();
        break;
    case eIntegrationScheme::TRI_POSITION_BASED_DYNAMIC:
    case eIntegrationScheme::TRI_BARAFF:
        scene = std::make_shared<cTrimeshScene>();
        break;
    default:
        SIM_ERROR("unsupported sim scene {}", type);
        break;
    }
    return scene;
}