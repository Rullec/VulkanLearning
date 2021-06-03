#pragma once
#include "utils/DefUtil.h"
#include "utils/MathUtil.h"
struct tVertex;
struct tEdge;
struct tTriangle;
struct tCollisionPoint
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    // int mBody0, mBody1;
    // enum mBody0Type, mBody1Type;
    // void *mBody0Ptr, *mBody1Ptr;
    tCollisionPoint();
    tVector
        mContactPt; // we think the contact pair are located in the same point
    tVector mContactNormal; // contact normal point from body1 to body0
    int mVertexId;          // the contact vertex id on the cloth
    double mPenetration;    // negative means there is a penetration

    // in the most simple case now, the normal pointed to the cloth
};

SIM_DECLARE_PTR(tCollisionPoint);
/**
 * \brief       Execute the collision detection
 */
class cCollisionDetecter
{
public:
    cCollisionDetecter();
    void AddCloth(std::vector<tVertex *> *vertex_array,
                  std::vector<tEdge *> *edge_array,
                  std::vector<tTriangle *> *tri_array);
    void PerformCollisionDetect();
    void Clear();
    const tEigenArr<tCollisionPointPtr> &GetCollisionPoints() const;

protected:
    std::vector<tVertex *> *mVertexArray;
    std::vector<tEdge *> *mEdgeArray;
    std::vector<tTriangle *> *mTriangleArray;
    tEigenArr<tCollisionPointPtr> mColPoints;
};