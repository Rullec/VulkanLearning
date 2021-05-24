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
    explicit cRaycaster();
    virtual void AddResources(const std::vector<tTriangle *> triangles,
                              const std::vector<tVertex *> vertices);
    void RayCast(const tRay *ray, tTriangle **selected_tri,
                 int &selected_tri_id, tVector &raycast_point) const;
    virtual void CalcDepthMap(int height, int width, CameraBasePtr camera,
                              std::string path);

protected:
    std::vector<std::vector<tTriangle *>> mTriangleArray_lst;
    std::vector<std::vector<tVertex *>> mVertexArray_lst;
};