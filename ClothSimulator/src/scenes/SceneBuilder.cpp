#include "SceneBuilder.h"
#include "PDScene.h"
#include "ImplicitScene.h"
#include "SemiImplicitScene.h"
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
    case eIntegrationScheme::MS_IMPLICIT:
        SIM_ERROR("hasn't been impled yet");
        break;
    case eIntegrationScheme::PROJECTIVE_DYNAMIC:
        scene = std::make_shared<cPDScene>();
        break;
    case eIntegrationScheme::MS_SEMI_IMPLICIT:
        scene = std::make_shared<cSemiImplicitScene>();
        break;
    case eIntegrationScheme::TRI_POSITION_BASED_DYNAMIC:
        scene = std::make_shared<cPBDScene>();
        break;
    case eIntegrationScheme::TRI_BARAFF:
        SIM_ERROR("hasn't been impled yet");
    default:
        SIM_ERROR("unsupported sim scene {}", type);
        break;
    }
    return scene;
}