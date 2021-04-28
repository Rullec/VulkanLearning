#include "Raycaster.h"
#include "utils/LogUtil.h"
cRaycaster::cRaycaster(const std::vector<tTriangle *> *triangles,
                       const std::vector<tVertex *> *vertices) : mTriangleArray(triangles), mVertexArray(vertices)
{
    SIM_INFO("Build raycaster succ");
    SIM_ASSERT(triangles != nullptr);
}

/**
 * \brief               Do raycast, calculate the intersection
*/
void cRaycaster::RayCast(const tRay *ray, tTriangle **selected_tri,
                         int &selected_tri_id,
                         tVector &raycast_point) const
{
    // 1. init
    *selected_tri = nullptr;
    selected_tri_id = -1;
    double min_depth = std::numeric_limits<double>::max();
    raycast_point.noalias() = tVector::Ones() * std::nan("");

    // 2. iterate on each triangle
    for (int i = 0; i < mTriangleArray->size(); i++)
    {
        auto &tri = mTriangleArray->at(i);
        tVector tmp = cMathUtil::RayCastTri(
            ray->mOrigin, ray->mDir, mVertexArray->at(tri->mId0)->mPos,
            mVertexArray->at(tri->mId1)->mPos, mVertexArray->at(tri->mId2)->mPos);

        // if there is an intersection, tmp will have no nan
        if (tmp.hasNaN() == false)
        {
            // std::cout << tmp.transpose() << std::endl;
            double cur_depth = (tmp - ray->mOrigin).segment(0, 3).norm();
            if (cur_depth < min_depth)
            {
                *selected_tri = tri;
                selected_tri_id = i;
                min_depth = cur_depth;
                raycast_point = tmp;
            }
        }
    }
}

/**
 * \brief           Calculate the depth image
*/
void cRaycaster::CalcDepthMap(int height, int width, CameraBasePtr camera)
{
    SIM_ASSERT("cRaycaster::CalcDepthMap hasn't been finished yet");
}