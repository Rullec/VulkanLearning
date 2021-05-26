#include "Raycaster.h"
#include "utils/LogUtil.h"
// cRaycaster::cRaycaster(const std::vector<tTriangle *> triangles,
//                        const std::vector<tVertex *> vertices) :
//                        mTriangleArray(triangles), mVertexArray(vertices)
cRaycaster::cRaycaster(bool enable_only_export_cutted_window)
    : mEnableOnlyExportCuttedWindow(enable_only_export_cutted_window)
{
    mTriangleArray_lst.clear();
    mVertexArray_lst.clear();
    // SIM_INFO("Build raycaster succ");
    // SIM_ASSERT(triangles != nullptr);
}

void cRaycaster::AddResources(const std::vector<tTriangle *> triangles,
                              const std::vector<tVertex *> vertices)
{
    mTriangleArray_lst.push_back(triangles);
    mVertexArray_lst.push_back(vertices);
}

/**
 * \brief               Do raycast, calculate the intersection
 */
void cRaycaster::RayCast(const tRay *ray, tTriangle **selected_tri,
                         int &selected_tri_id, tVector &raycast_point) const
{
    // 1. init
    *selected_tri = nullptr;
    selected_tri_id = -1;
    double min_depth = std::numeric_limits<double>::max();
    raycast_point.noalias() = tVector::Ones() * std::nan("");

    // 2. iterate on each triangle
    for (int obj_id = 0; obj_id < mTriangleArray_lst.size(); obj_id++)
    {
        auto mTriangleArray = mTriangleArray_lst[obj_id];
        auto mVertexArray = mVertexArray_lst[obj_id];
        for (int i = 0; i < mTriangleArray.size(); i++)
        {
            auto &tri = mTriangleArray[i];
            tVector tmp = cMathUtil::RayCastTri(
                ray->mOrigin, ray->mDir, mVertexArray[tri->mId0]->mPos,
                mVertexArray[tri->mId1]->mPos, mVertexArray[tri->mId2]->mPos);

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
}

/**
 * \brief           Calculate the depth image
 */
void cRaycaster::CalcDepthMap(const tMatrix2i &cast_range, int height,
                              int width, CameraBasePtr camera, std::string)
{
    SIM_ASSERT("cRaycaster::CalcDepthMap hasn't been finished yet");
}

/**
 * \brief           Calculate the interested window size
 */
void cRaycaster::CalcCastWindowSize(const tMatrix2i &cast_range_window,
                                    int &window_width, int &window_height,
                                    tVector2i &st)
{
    window_width = cast_range_window(0, 1) - cast_range_window(0, 0);
    window_height = cast_range_window(1, 1) - cast_range_window(1, 0);

    st = cast_range_window.col(0).transpose();
}