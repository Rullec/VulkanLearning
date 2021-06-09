#pragma once
#include "utils/DefUtil.h"
SIM_DECLARE_CLASS_AND_PTR(cBaseCloth)
namespace Json
{
class Value;
};
cBaseClothPtr BuildCloth(Json::Value conf);
