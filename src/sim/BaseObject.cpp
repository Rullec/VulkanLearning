#include "BaseObject.h"
#include "utils/LogUtil.h"
#include "utils/MathUtil.h"
#include <string>
std::string gObjectTypeStr[eObjectType::NUM_OBJ_TYPES] = {"Rigidbody", "Cloth"};

cBaseObject::cBaseObject(eObjectType type) : mType(type) {}

cBaseObject::~cBaseObject() {}

eObjectType cBaseObject::BuildObjectType(std::string str)
{
    eObjectType type = eObjectType::INVALID_OBJ_TYPE;
    for (int i = 0; i < eObjectType::NUM_OBJ_TYPES; i++)
    {
        if (gObjectTypeStr[i] == str)
        {
            type = static_cast<eObjectType>(i);
            break;
        }
    }

    SIM_ASSERT(type != eObjectType::INVALID_OBJ_TYPE);
    return type;
}
