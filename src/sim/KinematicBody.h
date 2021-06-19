#include "sim/BaseObject.h"
#include "utils/MathUtil.h"

struct tTriangle;
struct tEdge;
struct tVertex;
enum eKinematicBodyShape
{
    KINEMATIC_PLANE = 0,
    KINEMATIC_CUBE,
    KINEMATIC_SPHERE,
    KINEMATIC_CAPSULE,
    KINEMATIC_CUSTOM,
    NUM_OF_KINEMATIC_SHAPE,
    KINEMATIC_INVALID
};

class cKinematicBody : public cBaseObject
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    inline const static std::string TYPE_KEY = "type",
                                    MESH_PATH_KEY = "mesh_path",
                                    TARGET_AABB_KEY = "target_aabb",
                                    TRANSLATION_KEY = "translation",
                                    ORIENTATION_KEY = "orientation",
                                    PLANE_EQUATION_KEY = "equation",
                                    PLANE_SCALE_KEY = "plane_scale";
    cKinematicBody(int id_);
    virtual ~cKinematicBody();
    virtual void Init(const Json::Value &conf) override;
    static eKinematicBodyShape BuildKinematicBodyShape(std::string type_str);
    bool IsStatic() const;
    eKinematicBodyShape GetBodyShape() const;

    virtual void CalcTriangleDrawBuffer(Eigen::Map<tVectorXf> &res,
                                        int &st) const override final;
    virtual void CalcEdgeDrawBuffer(Eigen::Map<tVectorXf> &res,
                                    int &st) const override final;

    // virtual void UpdatePos(double dt) override final;
    // virtual void UpdateRenderingResource() override final;
    virtual tMatrix GetWorldTransform() const; //
protected:
    eKinematicBodyShape mBodyShape;
    std::string mCustomMeshPath;
    tVector mTargetAABB;
    tVector mInitPos;
    tVector mInitOrientation;
    bool mIsStatic;
    tVector mPlaneEquation;
    double mPlaneScale;
    // methods
    void BuildCustomKinematicBody();
    void BuildPlane();
    // virtual void InitDrawBuffer() override final;
};