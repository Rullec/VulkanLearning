#include "utils/DefUtil.h"

SIM_DECLARE_CLASS_AND_PTR(cKinematicBody);
namespace Json
{
class Value;
};
cKinematicBodyPtr BuildKinematicBody(const Json::Value &conf, int id_);