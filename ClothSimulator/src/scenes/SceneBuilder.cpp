#include "SceneBuilder.h"
#include "utils/LogUtil.h"

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