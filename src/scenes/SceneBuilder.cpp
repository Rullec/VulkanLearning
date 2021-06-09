#include "SceneBuilder.h"
#include "DrawScene.h"
#include "LinctexScene.h"
// #include "MeshVisScene.h"
// #include "ProcessTrainDataScene.h"
// #include "SynDataScene.h"
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
    case eSceneType::SCENE_SIM:
        scene = std::make_shared<cSimScene>();
        break;
#ifdef _WIN32
    case eSceneType::SCENE_SE:
        scene = std::make_shared<cLinctexScene>();
        break;
        // case eSceneType::SCENE_SYN_DATA:
        //     scene = std::make_shared<cSynDataScene>();
        //     break;
        // case eSceneType::SCENE_PROCESS_DATA:
        //     scene = std::make_shared<cProcessTrainDataScene>();
        //     break;
#endif
    // case eSceneType::SCENE_MESH_VIS:
    //     scene = std::make_shared<cMeshVisScene>();
    //     break;
    default:
        SIM_ERROR("unsupported sim scene {}", type);
        break;
    }
    return scene;
}