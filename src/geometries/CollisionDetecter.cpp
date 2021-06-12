#include "CollisionDetecter.h"
#include "geometries/CollisionInfo.h"
#include "sim/BaseObject.h"
#include "sim/KinematicBody.h"
#include "sim/cloth/BaseCloth.h"
#include "utils/LogUtil.h"
#include "utils/TimeUtil.hpp"
#include <iostream>
cCollisionDetecter::cCollisionDetecter() {}
cCollisionDetecter::~cCollisionDetecter() {}

void cCollisionDetecter::AddObject(cBaseObjectPtr obj,
                                   bool enable_self_collision /* = false*/)
{
    mColObjs.push_back(obj);
    mEnableSelfCollision.push_back(enable_self_collision);
}

void cCollisionDetecter::PerformCD()
{
    cTimeUtil::Begin("DCD");
    // ! 1. clear all buffers
    Clear();

    // ! 2. broadphase collision detection
    BroadphaseCD();

    // ! 3. narrowphase collision detection
    NarrowphaseCD();
    cTimeUtil::End("DCD");
}

void cCollisionDetecter::Clear()
{
    mContactPoints.clear();

    mColCandiadatePairs.clear();
}

std::vector<tColPointPtr> cCollisionDetecter::GetContactPoints() const
{
    return mContactPoints;
}

/**
 * \brief           do broadphase collision (judge AABB intersection)
 */
void cCollisionDetecter::BroadphaseCD()
{
    SIM_ASSERT(this->mColCandiadatePairs.size() == 0);
    // 1. calculate AABB
    int num_of_obj = mColObjs.size();
    tEigenArr<tVector> mAABBmin(num_of_obj), mAABBmax(num_of_obj);

    for (int i = 0; i < num_of_obj; i++)
    {
        mColObjs[i]->CalcAABB(mAABBmin[i], mAABBmax[i]);
    }

    // 2. compare
    for (int i = 0; i < num_of_obj; i++)
    {
        auto obj_type_i = mColObjs[i]->GetObjectType();
        for (int j = i + 1; j < num_of_obj; j++)
        {
            if (true == cMathUtil::IntersectAABB(mAABBmin[i], mAABBmax[i],
                                                 mAABBmin[j], mAABBmax[j]))
            {
                auto obj_type_j = mColObjs[j]->GetObjectType();
                if (obj_type_i <= obj_type_j)
                {

                    mColCandiadatePairs.push_back(tVector2i(i, j));
                }
                else
                {
                    mColCandiadatePairs.push_back(tVector2i(j, i));
                }
                // printf("[debug] broadphse %d and %d collided\n", i, j);
            }
        }
    }
}

/**
 * \brief           do narrowphase collision (no self collision)
 */
void cCollisionDetecter::NarrowphaseCD()
{
    // ! if the broadphase is empty, return
    if (mColCandiadatePairs.size() == 0)
        return;

    // ! begin to check eacy candidate collided obj pair
    for (auto &pair : mColCandiadatePairs)
    {
        int obj0_id = pair[0], obj1_id = pair[1];
        eObjectType type0 = mColObjs[obj0_id]->GetObjectType(),
                    type1 = mColObjs[obj1_id]->GetObjectType();
        // type0 <= type1 is guranteed by the broadphase
        // std::cout << "type0 = " << type0 << std::endl;
        // std::cout << "type1 = " << type1 << std::endl;
        if ((type0 == eObjectType::KINEMATICBODY_TYPE) &&
            (type1 == eObjectType::CLOTH_TYPE))
        {
            auto rb_ptr =
                std::dynamic_pointer_cast<cKinematicBody>(mColObjs[obj0_id]);
            auto cloth_ptr =
                std::dynamic_pointer_cast<cBaseCloth>(mColObjs[obj1_id]);
            ClothRigidBodyIntersection(cloth_ptr, rb_ptr);
        }
    }
}

/**
 * \brief           do cloth - rigidbody intersection
 */
void cCollisionDetecter::ClothRigidBodyIntersection(cBaseClothPtr cloth,
                                                    cKinematicBodyPtr rb)
{
    auto TrianglePointerInersection =
        [](tTriangle *tri, std::vector<tVertex *> tri_vertex_array, tVertex *v,
           double &depth, tVector &normal)
    {
        // 1. get face normal
        const tVector p0 = tri_vertex_array[tri->mId0]->mPos;
        const tVector p1 = tri_vertex_array[tri->mId1]->mPos;
        const tVector p2 = tri_vertex_array[tri->mId2]->mPos;
        normal = (p1 - p0).cross3(p2 - p1);
        normal[3] = 0;
        normal.normalize();
        // 2. get face -> cloth vertex vector
        tVector to_cloth = v->mPos - p0;
        to_cloth[3] = 0;
        depth = to_cloth.dot(normal);

        // 3. oppo means there maybe an interseciton
        if (depth < 0)
        {
            // ray - triangle intersection
            tVector pos = cMathUtil::RayCastTri(v->mPos, normal, p0, p1, p2);

            // there is an intersection
            if (pos.hasNaN() == false)
                return true;
        }

        // there is no intersection
        return false;
    };
    // ! 1. now we only do vertex and face intersection
    int num_cloth_vertices = cloth->GetNumOfVertices(),
        num_rb_vertices = rb->GetNumOfVertices();

    // ! 1.1 cloth vertex and rb face
    const auto &cloth_v_array = cloth->GetVertexArray(),
               rb_v_array = rb->GetVertexArray();
    const auto &cloth_tri_array = cloth->GetTriangleArray(),
               rb_tri_array = rb->GetTriangleArray();

    // for (auto cloth_v : cloth_v_array)
    for (int v_id = 0; v_id < cloth_v_array.size(); v_id++)
    {
        auto cloth_v = cloth_v_array[v_id];
        for (auto rb_face : rb_tri_array)
        {
            double depth = 0;
            tVector face_normal = tVector::Zero();
            if (true == TrianglePointerInersection(rb_face, rb_v_array, cloth_v,
                                                   depth, face_normal))
            {
                /*
                    obj0: cloth
                    obj1: rigidbody

                    normal: from cloth to rigidbody
                */
                // calculate rb local pos
                tVector rb_collision_local_pos = tVector::Zero();
                {
                    // local_pos = rb_trans.inv() * world_pos
                    tVector rb_collision_world_pos =
                        cloth_v->mPos + face_normal * depth;
                    rb_collision_world_pos[3] = 1;
                    rb_collision_local_pos = rb->GetWorldTransform().inverse() *
                                             rb_collision_world_pos;
                }
                // there is an interseciton!
                tColPointPtr col_pt = std::make_shared<tColPoint>();
                col_pt->mDepth = depth;
                col_pt->mNormal = -face_normal; // from obj0 to obj1
                col_pt->mObjInfo0 =
                    std::make_shared<tColClothInfo>(cloth, v_id);
                col_pt->mObjInfo1 = std::make_shared<tColRigidBodyInfo>(
                    rb, rb_collision_local_pos);
                this->mContactPoints.push_back(col_pt);
            }
        }
    }
    // ! 1.2 rb vertex and cloth face

    // printf("[debug] narrow phase %d contacts\n", mContactPoints.size());
}