#include "SceneBuilder.h"
#include "PDScene.h"
#include "ImplicitScene.h"
#include "SemiImplicitScene.h"
#include "BaraffScene.h"
#include "PBDScene.h"
#include "utils/JsonUtil.h"
#include "utils/LogUtil.h"

std::shared_ptr<cDrawScene>
cSceneBuilder::BuildScene(const std::string type, bool enable_draw /*= true*/)
{
    std::shared_ptr<cDrawScene> scene = nullptr;
    if (enable_draw == true)
    {
        scene = std::make_shared<cDrawScene>();
    }
    else
    {
        SIM_ASSERT(false);
    }
    return scene;
}
std::shared_ptr<cSimScene>
cSceneBuilder::BuildSimScene(const std::string config_file)
{
    Json::Value root;
    cJsonUtil::LoadJson(config_file, root);
    std::string type = cJsonUtil::ParseAsString("integration_scheme", root);
    eIntegrationScheme scheme = cSimScene::BuildIntegrationScheme(type);
    std::shared_ptr<cSimScene> scene = nullptr;
    switch (scheme)
    {
    case eIntegrationScheme::SCHEME_IMPLICIT:
        scene = std::make_shared<cImplicitScene>();
        break;
    case eIntegrationScheme::SCHEME_PROJECTIVE_DYNAMIC:
        scene = std::make_shared<cPDScene>();
        break;
    case eIntegrationScheme::SCHEME_SEMI_IMPLICIT:
        scene = std::make_shared<cSemiImplicitScene>();
        break;
    case eIntegrationScheme::SCHEME_POSITION_BASED_DYNAMIC:
        scene = std::make_shared<cPBDScene>();
        break;
    case eIntegrationScheme::SCHEME_BARAFF:
        scene = std::make_shared<cBaraffScene>();
        break;
    default:
        SIM_ERROR("unsupported sim scene {}", type);
        break;
    }
    return scene;
}