#include "KinematicBodyBuilder.h"
#include "KinematicBody.h"

cKinematicBodyPtr BuildKinematicBody(const Json::Value &conf, int id)
{
    cKinematicBodyPtr ptr = std::make_shared<cKinematicBody>(id);
    ptr->Init(conf);
    return ptr;
}