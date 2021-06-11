#include "CollisionManager.h"
#include "sim/BaseObject.h"
#include "utils/LogUtil.h"
cCollisionManager::cCollisionManager() {}

void cCollisionManager::AddObject(cBaseObjectPtr obj,
                                  bool enable_self_collision /* = false*/)
{
    mColObjs.push_back(obj);
    mEnableSelfCollision.push_back(enable_self_collision);
}

void cCollisionManager::PerformCD()
{
    // ! 1. clear all buffers
    Clear();

    // ! 2. broadphase collision detection
    BroadphaseCD();

    // ! 3. narrowphase collision detection
    NarrowphaseCD();
}

void cCollisionManager::Clear()
{
    mContactPoints.clear();

    mColCandiadatePairs.clear();
}

std::vector<tColPointPtr> cCollisionManager::GetContactPoints() const
{
    return mContactPoints;
}

/**
 * \brief           do broadphase collision (judge AABB intersection)
 */
void cCollisionManager::BroadphaseCD()
{
    SIM_ASSERT(this->mColCandiadatePairs.size() > 0);
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
        for (int j = i + 1; j < num_of_obj; j++)
        {
            if (true == cMathUtil::IntersectAABB(mAABBmin[i], mAABBmax[i],
                                                 mAABBmin[j], mAABBmax[j]))
            {
                mColCandiadatePairs.push_back(tVector2i(i, j));
                printf("[debug] broadphse %d and %d collided\n", i, j);
            }
        }
    }
}

/**
 * \brief           do narrowphase collision (no self collision)
 */
void cCollisionManager::NarrowphaseCD()
{
    // ! if the broadphase is empty, return
    if (mColCandiadatePairs.size() == 0)
        return;

    // ! begin to check eacy candidate collided obj pair
    for (auto &pair : mColCandiadatePairs)
    {
    }
}

/**
 * \brief           do cloth - rigidbody intersection
 */
void cCollisionManager::ClothRigidBodyIntersection(cBaseClothPtr cloth,
                                                   cKinematicBodyPtr rb)
{
    // ! 1. now we do not vertex and face intersection
}