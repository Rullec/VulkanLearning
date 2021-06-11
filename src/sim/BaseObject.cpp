#include "sim/BaseObject.h"
#include "geometries/Primitives.h"
#include "utils/LogUtil.h"
#include "utils/MathUtil.h"
#include <string>
std::string gObjectTypeStr[eObjectType::NUM_OBJ_TYPES] = {
    "KinematicBody", "RigidBody", "Cloth", "Fluid"};

cBaseObject::cBaseObject(eObjectType type) : mType(type)
{
    mEnableDrawBuffer = true;
}

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

eObjectType cBaseObject::GetObjectType() const { return this->mType; }

int cBaseObject::GetNumOfTriangles() const { return mTriangleArray.size(); }
int cBaseObject::GetNumOfEdges() const { return mEdgeArray.size(); }
int cBaseObject::GetNumOfVertices() const { return mVertexArray.size(); }
const std::vector<tVertex *> &cBaseObject::GetVertexArray() const
{
    return this->mVertexArray;
}
const std::vector<tEdge *> &cBaseObject::GetEdgeArray() const
{
    return this->mEdgeArray;
}
const std::vector<tTriangle *> &cBaseObject::GetTriangleArray() const
{
    return this->mTriangleArray;
}

void cBaseObject::ChangeTriangleColor(int tri_id, tVector color)
{
    mVertexArray[mTriangleArray[tri_id]->mId0]->mColor = color;
    mVertexArray[mTriangleArray[tri_id]->mId1]->mColor = color;
    mVertexArray[mTriangleArray[tri_id]->mId2]->mColor = color;
}