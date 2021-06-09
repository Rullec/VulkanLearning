#include "sim/cloth/ClothBuilder.h"
#include "sim/cloth/BaseCloth.h"
#include "sim/cloth/ImplicitCloth.h"
#include "sim/cloth/PBDCloth.h"
#include "sim/cloth/PDCloth.h"
#include "sim/cloth/SemiCloth.h"
#include "utils/JsonUtil.h"
#include <iostream>
#include <string>

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
    case eClothType::PBD_CLOTH:
        ptr = std::make_shared<cPBDCloth>();
        break;
    case eClothType::IMPLICIT_CLOTH:
        ptr = std::make_shared<cImplicitCloth>();
        break;
    case eClothType::PD_CLOTH:
        ptr = std::make_shared<cPDCloth>();
        break;
    default:
        std::cout << "pbd type = " << eClothType::PBD_CLOTH
                  << " cur type = " << type << std::endl;
        SIM_ERROR("unsupported cloth type enum {}", type);
        break;
    }
    return ptr;
}