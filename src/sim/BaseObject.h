#pragma once

enum eObjectType
{
    RIGIDBODY_TYPE,
    CLOTH_TYPE,
    NUM_OBJ_TYPES,
    INVALID_OBJ_TYPE
}

/**
 * \brief           base object class
 * 
*/
class cBaseObject
{
public:
    cBaseObject(eObjectType type);
    ~cBaseObject();
    virtual void Init() = 0;
    static eObjectType BuildObjectType(std::string type);

protected:
    eObjectType mType;
};
