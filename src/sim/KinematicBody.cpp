#include "KinematicBody.h"
#include "utils/JsonUtil.h"
#include "utils/ObjUtil.h"
#include <iostream>
std::string gBodyShapeStr[eKinematicBodyShape::NUM_OF_KINEMATIC_SHAPE] = {
    "plane", "cube", "sphere", "capsule", "custom"};

cKinematicBody::cKinematicBody(int id_)
    : cBaseObject(eObjectType::KINEMATICBODY_TYPE, id_)
{
    mIsStatic = true;
    mBodyShape = eKinematicBodyShape::KINEMATIC_INVALID;
    mCustomMeshPath = "";
    mTargetAABB = tVector::Zero();
    mPlaneEquation.setZero();
}

cKinematicBody::~cKinematicBody() {}
void cKinematicBody::Init(const Json::Value &value)
{
    cBaseObject::Init(value);
    std::string type =
        cJsonUtil::ParseAsString(cKinematicBody::TYPE_KEY, value);
    mBodyShape = BuildKinematicBodyShape(type);
    switch (mBodyShape)
    {
    case eKinematicBodyShape::KINEMATIC_CUSTOM:
    {
        mCustomMeshPath =
            cJsonUtil::ParseAsString(cKinematicBody::MESH_PATH_KEY, value);
        cJsonUtil::ReadVectorJson(
            cJsonUtil::ParseAsValue(cKinematicBody::TARGET_AABB_KEY, value),
            mTargetAABB);
        cJsonUtil::ReadVectorJson(
            cJsonUtil::ParseAsValue(cKinematicBody::TRANSLATION_KEY, value),
            mInitPos);
        cJsonUtil::ReadVectorJson(
            cJsonUtil::ParseAsValue(cKinematicBody::ORIENTATION_KEY, value),
            mInitOrientation);
        // std::cout << "target aabb = " << mTargetAABB.transpose() <<
        // std::endl;
        BuildCustomKinematicBody();
        break;
    }
    case eKinematicBodyShape::KINEMATIC_PLANE:
    {
        cJsonUtil::ReadVectorJson(
            cJsonUtil::ParseAsValue(cKinematicBody::PLANE_EQUATION_KEY, value),
            mPlaneEquation);
        // std::cout << "plane equation = " << mPlaneEquation.transpose() <<
        // std::endl;
        mPlaneScale =
            cJsonUtil::ParseAsDouble(cKinematicBody::PLANE_SCALE_KEY, value);
        BuildPlane();
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

eKinematicBodyShape
cKinematicBody::BuildKinematicBodyShape(std::string type_str)
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

bool cKinematicBody::IsStatic() const { return mIsStatic; }
#include "geometries/Triangulator.h"

/**
 * \brief           Build plane data strucutre
 */
void cKinematicBody::BuildPlane()
{
    // 1. build legacy XOZ plane, then do a transformation
    // for (int i = 0; i < 4; i++)
    cObjUtil::BuildPlaneGeometryData(mPlaneScale, this->mPlaneEquation,
                                     mVertexArray, mEdgeArray, mTriangleArray);
    for (auto &x : mVertexArray)
        x->mColor = tVector(0.1f, 0.1f, 0.1f, 0);
}

/**
 * \brief               Build custom kinematic body
 */
void cKinematicBody::BuildCustomKinematicBody()
{
    // std::cout << "mesh path = " << mCustomMeshPath << std::endl;
    cObjUtil::tParams obj_params;
    obj_params.mPath = mCustomMeshPath;
    cObjUtil::LoadObj(obj_params, mVertexArray, mEdgeArray, mTriangleArray);

    tMatrix trans = GetWorldTransform();
    tVector aabb_min, aabb_max;
    CalcAABB(aabb_min, aabb_max);
    tVector aabb = aabb_max - aabb_min;
    // std::cout << "init aabb = " << (aabb_max - aabb_min).transpose() <<
    // std::endl; exit(0);
    tMatrix scale_mat = tMatrix::Identity();
    for (int i = 0; i < 3; i++)
    {
        scale_mat(i, i) = mTargetAABB[i] / aabb[i];
    }
    // scale_mat.topLeftCorner<3, 3>() *= mScale;
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
    cTriangulator::ValidateGeometry(mVertexArray, mEdgeArray, mTriangleArray);
}

// int cKinematicBody::GetDrawNumOfTriangles() const
// {
//     return mTriangleArray.size();
// }

// int cKinematicBody::GetDrawNumOfEdges() const { return mEdgeArray.size(); }
extern void CalcTriangleDrawBufferSingle(tVertex *v0, tVertex *v1, tVertex *v2,
                                         Eigen::Map<tVectorXf> &buffer,
                                         int &st_pos);
extern void CalcEdgeDrawBufferSingle(tVertex *v0, tVertex *v1,
                                     Eigen::Map<tVectorXf> &buffer,
                                     int &st_pos);

void cKinematicBody::CalcTriangleDrawBuffer(Eigen::Map<tVectorXf> &res,
                                            int &st) const
{
    for (auto &x : mTriangleArray)
    {
        CalcTriangleDrawBufferSingle(mVertexArray[x->mId0],
                                     mVertexArray[x->mId1],
                                     mVertexArray[x->mId2], res, st);
    }
}

void cKinematicBody::CalcEdgeDrawBuffer(Eigen::Map<tVectorXf> &res,
                                        int &st) const
{
    for (auto &x : mEdgeArray)
    {
        CalcEdgeDrawBufferSingle(mVertexArray[x->mId0], mVertexArray[x->mId1],
                                 res, st);
    }
}

/**
 * \brief           Get the world transform of this kinematic body
 *          this matrix can convert local pos to world pos in homogeneous coords
 *          world_pos = T * local_pos
 */
tMatrix cKinematicBody::GetWorldTransform() const
{
    tMatrix trans = tMatrix::Identity();
    trans.block(0, 3, 3, 1) = mInitPos.segment(0, 3);
    trans.block(0, 0, 3, 3) =
        cMathUtil::EulerAnglesToRotMat(mInitOrientation, eRotationOrder::XYZ)
            .topLeftCorner<3, 3>();
    return trans;
}