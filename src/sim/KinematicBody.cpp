#include "KinematicBody.h"
#include "utils/ObjUtil.h"
#include "utils/JsonUtil.h"
#include <iostream>
std::string gBodyShapeStr[eKinematicBodyShape::NUM_OF_KINEMATIC_SHAPE] = {
    "plane",
    "cube",
    "sphere",
    "capsule",
    "custom"};

cKinematicBody::cKinematicBody() : cBaseObject(eObjectType::KINEMATICBODY_TYPE)
{
    mIsStatic = true;
    mBodyShape = eKinematicBodyShape::KINEMATIC_INVALID;
    mCustomMeshPath = "";
}

cKinematicBody::~cKinematicBody()
{
}
void cKinematicBody::Init(const Json::Value &value)
{
    std::string type = cJsonUtil::ParseAsString("type", value);
    mBodyShape = BuildKinematicBodyShape(type);
    switch (mBodyShape)
    {
    case eKinematicBodyShape::KINEMATIC_CUSTOM:
    {
        mCustomMeshPath = cJsonUtil::ParseAsString("mesh_path", value);
        mScale = cJsonUtil::ParseAsDouble("scale", value);
        cJsonUtil::ReadVectorJson(
            cJsonUtil::ParseAsValue("position", value), mInitPos);
        cJsonUtil::ReadVectorJson(
            cJsonUtil::ParseAsValue("orientation", value), mInitOrientation);
        // std::cout << "init scale = " << mScale << std::endl;
        // std::cout << "init position = " << mInitPos.transpose() << std::endl;
        // std::cout << "init orientation = " << mInitOrientation.transpose() << std::endl;
        // exit(0);
        BuildCustomKinematicBody();
        break;
    }

    default:
        SIM_ERROR("Unsupported kinematic shape {}", type);
    }
    tVector min, max;
    CalcAABB(min, max);
    std::cout << "[debug] obstacle aabb min = " << min.transpose() << std::endl;
    std::cout << "[debug] obstacle aabb max = " << max.transpose() << std::endl;
}

eKinematicBodyShape cKinematicBody::BuildKinematicBodyShape(std::string type_str)
{
    eKinematicBodyShape shape = eKinematicBodyShape::KINEMATIC_INVALID;
    for (int i = 0; i < eKinematicBodyShape::NUM_OF_KINEMATIC_SHAPE; i++)
    {
        if (gBodyShapeStr[i] == type_str)
        {
            shape = static_cast<eKinematicBodyShape>(i);
            break;
        }
    }
    SIM_ASSERT(shape != eKinematicBodyShape::KINEMATIC_INVALID);
    return shape;
}

bool cKinematicBody::IsStatic() const
{
    return mIsStatic;
}
#include "geometries/Triangulator.h"
void cKinematicBody::BuildCustomKinematicBody()
{
    // std::cout << "mesh path = " << mCustomMeshPath << std::endl;
    cObjUtil::tParams obj_params;
    obj_params.mPath = mCustomMeshPath;
    cObjUtil::LoadObj(obj_params,
                      mVertexArray,
                      mEdgeArray,
                      mTriangleArray);

    tMatrix trans = tMatrix::Identity();
    {
        trans.block(0, 3, 3, 1) = mInitPos.segment(0, 3);
        trans.block(0, 0, 3, 3) = cMathUtil::EulerAnglesToRotMat(
                                      mInitOrientation, eRotationOrder::XYZ)
                                      .topLeftCorner<3, 3>();
    }
    tMatrix scale_mat = tMatrix::Identity();
    scale_mat.topLeftCorner<3, 3>() *= mScale;
    // std::cout << "trans = \n"
    //           << trans << std::endl;
    // std::cout << "init pos = " << mInitPos.transpose() << std::endl;
    // std::cout << "euler = " << mInitOrientation.transpose() << std::endl;
    // std::cout << "trans mat = \n"
    //           << trans << std::endl;
    // std::cout << "scale mat = \n"
    //           << scale_mat << std::endl;
    for (auto &x : mVertexArray)
    {
        x->mColor = tVector::Ones() * 0.3;
        // std::cout << "old pos = " << x->mPos.transpose() << std::endl;
        x->mPos = trans * scale_mat * (x->mPos);
        // std::cout << "new pos = " << x->mPos.transpose() << std::endl;
    }
    // exit(0);
    cTriangulator::ValidateGeometry(mVertexArray,
                                    mEdgeArray,
                                    mTriangleArray);
}

int cKinematicBody::GetDrawNumOfTriangles() const
{
    return mTriangleArray.size();
}

int cKinematicBody::GetDrawNumOfEdges() const
{
    return mEdgeArray.size();
}
extern void CalcTriangleDrawBufferSingle(tVertex *v0, tVertex *v1, tVertex *v2,
                                         Eigen::Map<tVectorXf> &buffer, int &st_pos);
extern void CalcEdgeDrawBufferSingle(tVertex *v0, tVertex *v1, Eigen::Map<tVectorXf> &buffer,
                                     int &st_pos);

void cKinematicBody::CalcTriangleDrawBuffer(Eigen::Map<tVectorXf> &res) const
{
    int st_pos = 0;
    for (auto &x : mTriangleArray)
    {
        CalcTriangleDrawBufferSingle(
            mVertexArray[x->mId0],
            mVertexArray[x->mId1],
            mVertexArray[x->mId2],
            res,
            st_pos);
    }
}

void cKinematicBody::CalcEdgeDrawBuffer(Eigen::Map<tVectorXf> &res) const
{
    int st_pos = 0;
    for (auto &x : mEdgeArray)
    {

        CalcEdgeDrawBufferSingle(
            mVertexArray[x->mId0],
            mVertexArray[x->mId1],
            res,
            st_pos);
    }
}
#include <cfloat>
void cKinematicBody::CalcAABB(tVector &min, tVector &max) const
{
    min = tVector::Ones() * std::numeric_limits<double>::max();
    max = tVector::Ones() * std::numeric_limits<double>::max() * -1;
    for (auto &x : mVertexArray)
    {
        for (int i = 0; i < 3; i++)
        {

            double val = x->mPos[i];
            min[i] = (val < min[i]) ? val : min[i];
            max[i] = (val > max[i]) ? val : max[i];
        }
    }
}

const std::vector<tVertex *> &cKinematicBody::GetVertexArray() const
{
    return mVertexArray;
}
const std::vector<tEdge *> &cKinematicBody::GetEdgeArray() const
{
    return mEdgeArray;
}
const std::vector<tTriangle *> &cKinematicBody::GetTriangleArray() const
{
    return mTriangleArray;
}