#pragma once
#include "utils/MathUtil.h"
#include <memory>
#include <string>

/**
 * \brief           The ULTIMATE object type collections
 */
enum eObjectType
{
    KINEMATICBODY_TYPE,
    RIGIDBODY_TYPE,
    CLOTH_TYPE,
    FLUID_TYPE,
    NUM_OBJ_TYPES,
    INVALID_OBJ_TYPE
};

/**
 * \brief           base object class
 *
 */
namespace Json
{
class Value;
};

struct tVertex;
struct tEdge;
struct tTriangle;
class cBaseObject : std::enable_shared_from_this<cBaseObject>
{
public:
    explicit cBaseObject(eObjectType type);
    virtual ~cBaseObject();
    virtual void Init(const Json::Value &conf) = 0;
    static eObjectType BuildObjectType(std::string type);
    eObjectType GetObjectType() const;
    virtual void CalcTriangleDrawBuffer(Eigen::Map<tVectorXf> &res,
                                        int &st) const = 0;
    virtual void CalcEdgeDrawBuffer(Eigen::Map<tVectorXf> &res,
                                    int &st) const = 0;
    // virtual void Update(double dt) = 0;

    // triangularize methods to visit the mesh data
    virtual int GetNumOfTriangles() const;
    virtual int GetNumOfEdges() const;
    virtual int GetNumOfVertices() const;
    const std::vector<tVertex *> &GetVertexArray() const;
    const std::vector<tEdge *> &GetEdgeArray() const;
    const std::vector<tTriangle *> &GetTriangleArray() const;
    void ChangeTriangleColor(int tri_id, tVector color);
    virtual void CalcAABB(tVector &min, tVector &max) const;

protected:
    eObjectType mType;
    bool mEnableDrawBuffer; // enable to open draw buffer
    std::vector<tVertex *> mVertexArray;
    std::vector<tEdge *> mEdgeArray;
    std::vector<tTriangle *> mTriangleArray;
};
