#pragma once
#include "sim/BaseCloth.h"
#include "sim/SemiCloth.h"
#include "utils/DefUtil.h"
#include "utils/JsonUtil.h"
#include <string>
// #include "BaseCloth.h"
SIM_DECLARE_CLASS_AND_PTR(cBaseCloth)
namespace Json
{
class Value;
};
const std::string CLOTH_TYPE_KEY = "cloth_type";
cBaseClothPtr BuildCloth(Json::Value conf)
{
    std::string cloth_type_str = cJsonUtil::ParseAsString(CLOTH_TYPE_KEY, conf);
    eClothType type = cBaseCloth::BuildClothType(cloth_type_str);
    cBaseClothPtr ptr = nullptr;
    switch (type)
    {
    // case eClothType::IMPLICIT_CLOTH :
    //     S
    //     break;
    case eClothType::SEMI_IMPLICIT_CLOTH:
        ptr = std::make_shared<cSemiCloth>();
        break;
    default:
        SIM_ERROR("unsupported cloth type enum {}", type);
        break;
    }
    return ptr;
}