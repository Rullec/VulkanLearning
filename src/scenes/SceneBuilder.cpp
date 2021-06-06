#include "SceneBuilder.h"
#include "BaraffScene.h"
#include "DrawScene.h"
#include "ImplicitScene.h"
#include "LinctexScene.h"
#include "MeshVisScene.h"
#include "PBDScene.h"
#include "PDScene.h"
#include "ProcessTrainDataScene.h"
#include "SemiImplicitScene.h"
#include "SynDataScene.h"
#include "utils/JsonUtil.h"
#include "utils/LogUtil.h"

std::shared_ptr<cScene> cSceneBuilder::BuildScene(const std::string type,
                                                  bool enable_draw /*= true*/)
{
    std::shared_ptr<cScene> scene = nullptr;
    if (enable_draw == true)
    {
        scene =
            std::dynamic_pointer_cast<cScene>(std::make_shared<cDrawScene>());
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
    std::string type = cJsonUtil::ParseAsString("scene_type", root);
    eSceneType scheme = cSimScene::BuildSceneType(type);
    std::shared_ptr<cSimScene> scene = nullptr;
    switch (scheme)
    {
    case eSceneType::SCENE_IMPLICIT:
        scene = std::make_shared<cImplicitScene>();
        break;
    case eSceneType::SCENE_PROJECTIVE_DYNAMIC:
        scene = std::make_shared<cPDScene>();
        break;
    case eSceneType::SCENE_SEMI_IMPLICIT:
        scene = std::make_shared<cSemiImplicitScene>();
        break;
    case eSceneType::SCENE_POSITION_BASED_DYNAMIC:
        scene = std::make_shared<cPBDScene>();
        break;
    case eSceneType::SCENE_BARAFF:
        scene = std::make_shared<cBaraffScene>();
        break;
#ifdef _WIN32
    case eSceneType::SCENE_SE:
        scene = std::make_shared<cLinctexScene>();
        break;
    case eSceneType::SCENE_SYN_DATA:
        scene = std::make_shared<cSynDataScene>();
        break;
#endif
    case eSceneType::SCENE_PROCESS_DATA:
        scene = std::make_shared<cProcessTrainDataScene>();
        break;
    case eSceneType::SCENE_MESH_VIS:
        scene = std::make_shared<cMeshVisScene>();
        break;
    default:
        SIM_ERROR("unsupported sim scene {}", type);
        break;
    }
    return scene;
}