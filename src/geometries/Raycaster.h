#pragma once
#include <memory>
#include "Primitives.h"
#include "utils/DefUtil.h"

/**
 * \brief               do raycasting for the triangle-based scene
*/
SIM_DECLARE_CLASS_AND_PTR(CameraBase);
class cRaycaster : std::enable_shared_from_this<cRaycaster>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    explicit cRaycaster(const std::vector<tTriangle *> *triangles,
                        const std::vector<tVertex *> *vertices);
    void RayCast(const tRay *ray, tTriangle **selected_tri,
                 int &selected_tri_id,
                 tVector &raycast_point) const;
    virtual void CalcDepthMap(int height, int width, CameraBasePtr camera);

protected:
    const std::vector<tTriangle *> *mTriangleArray;
    const std::vector<tVertex *> *mVertexArray;
};