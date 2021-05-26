#pragma once
#include "Primitives.h"
#include "utils/DefUtil.h"
#include <memory>

/**
 * \brief               do raycasting for the triangle-based scene
 */
SIM_DECLARE_CLASS_AND_PTR(CameraBase);
class cRaycaster : std::enable_shared_from_this<cRaycaster>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    explicit cRaycaster(bool enable_only_exporting_cutted_window);
    virtual void AddResources(const std::vector<tTriangle *> triangles,
                              const std::vector<tVertex *> vertices);
    void RayCast(const tRay *ray, tTriangle **selected_tri,
                 int &selected_tri_id, tVector &raycast_point) const;
    virtual void CalcDepthMap(const tMatrix2i &cast_range_window, int height,
                              int width, CameraBasePtr camera,
                              std::string path);
    static void CalcCastWindowSize(const tMatrix2i &cast_range_window,
                                   int &window_width, int &window_height,
                                   tVector2i &st);

protected:
    std::vector<std::vector<tTriangle *>> mTriangleArray_lst;
    std::vector<std::vector<tVertex *>> mVertexArray_lst;
    bool mEnableOnlyExportCuttedWindow; // only save the interested window
                                        // result of raycasting
};