#pragma once
#include "utils/MathUtil.h"
#include <string>
enum eObjectType
{
    KINEMATICBODY_TYPE,
    RIGIDBODY_TYPE,
    CLOTH_TYPE,
    NUM_OBJ_TYPES,
    INVALID_OBJ_TYPE
};

/**
 * \brief           base object class
 *
 */
namespace Json
{
class Value;
};

class cBaseObject
{
public:
    explicit cBaseObject(eObjectType type);
    virtual ~cBaseObject();
    virtual void Init(const Json::Value &conf) = 0;
    static eObjectType BuildObjectType(std::string type);
    virtual int GetDrawNumOfTriangles() const = 0;
    virtual int GetDrawNumOfEdges() const = 0;
    virtual void CalcTriangleDrawBuffer(Eigen::Map<tVectorXf> &res) const = 0;
    virtual void CalcEdgeDrawBuffer(Eigen::Map<tVectorXf> &res) const = 0;

protected:
    eObjectType mType;
};
