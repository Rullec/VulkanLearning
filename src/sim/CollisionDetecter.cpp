#include "CollisionDetecter.h"
#include "geometries/Primitives.h"

tCollisionPoint::tCollisionPoint()
{
    mContactPt.setZero();
    mContactNormal.setZero();
    mVertexId = -1;
}

cCollisionDetecter::cCollisionDetecter()
{
    mVertexArray = nullptr;
    mEdgeArray = nullptr;
    mTriangleArray = nullptr;
    Clear();
}

/**
 * \brief           Add cloth into the collision detector
 */
void cCollisionDetecter::AddCloth(std::vector<tVertex *> *vertex_array,
                                  std::vector<tEdge *> *edge_array,
                                  std::vector<tTriangle *> *tri_array)
{
    mVertexArray = vertex_array;
    mEdgeArray = edge_array;
    mTriangleArray = tri_array;
}

/**
 * \brief           do collision detect
 */
void cCollisionDetecter::PerformCollisionDetect()
{
    Clear();
    double circle_radius = 0.05; // m, 5cm
    double height = 0.27;        // m, 27cm
    double threshold = 0.005;    // 5mm
    // triangle - cylinder intersection
    for (int t_id = 0; t_id < mVertexArray->size(); t_id++)
    {
        const tVector &pos = mVertexArray->at(t_id)->mPos;
        if (std::sqrt(std::pow(pos[0], 2) + std::pow(pos[2], 2)) <
                circle_radius &&
            pos[1] < height + threshold)
        {
            tCollisionPointPtr ptr = std::make_shared<tCollisionPoint>();
            ptr->mContactPt = pos;
            ptr->mContactNormal = tVector(0, 1, 0, 0);
            ptr->mVertexId = t_id;
            ptr->mPenetration = pos[1] - height;
            mColPoints.push_back(ptr);
        }
    }
}

/**
 * \brief           Clear all collision info
 * */
void cCollisionDetecter::Clear() { mColPoints.clear(); }

/**
 * \brief
 */
const tEigenArr<tCollisionPointPtr> &
cCollisionDetecter::GetCollisionPoints() const
{
    return mColPoints;
}
