#include "BaseObject.h"
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
    cKinematicBody();
    virtual ~cKinematicBody();
    virtual void Init(const Json::Value &conf) override;
    static eKinematicBodyShape BuildKinematicBodyShape(std::string type_str);
    bool IsStatic() const;
    eKinematicBodyShape GetBodyShape() const;

    virtual int GetDrawNumOfTriangles() const override final;
    virtual int GetDrawNumOfEdges() const override final;
    virtual void CalcTriangleDrawBuffer(Eigen::Map<tVectorXf> &res) const override final;
    virtual void CalcEdgeDrawBuffer(Eigen::Map<tVectorXf> &res) const override final;

    const std::vector<tVertex *> &GetVertexArray() const;
    const std::vector<tEdge *> &GetEdgeArray() const;
    const std::vector<tTriangle *> &GetTriangleArray() const;

protected:
    eKinematicBodyShape mBodyShape;
    std::string mCustomMeshPath;
    double mScale;
    tVector mInitPos;
    tVector mInitOrientation;
    bool mIsStatic;
    std::vector<tVertex *> mVertexArray;
    std::vector<tEdge *> mEdgeArray;
    std::vector<tTriangle *> mTriangleArray;

    // methods
    void BuildCustomKinematicBody();
    virtual void CalcAABB(tVector &min, tVector &max) const;
};