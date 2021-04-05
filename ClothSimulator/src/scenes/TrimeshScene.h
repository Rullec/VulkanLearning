#pragma once
#include "SimScene.h"

/**
 * \brief           using trimesh to modeling the cloth
 * 
 *  1. Positon based method 
 *  2. baraff 98 siggraph method
*/
struct tTriangle
{
    explicit tTriangle();
    explicit tTriangle(int a, int b, int c);
    int mId0, mId1, mId2;
};

struct tEdge
{
    tEdge();
    int mId0, mId1;
    double mRawLength;              // raw length of this edge
    bool mIsBoundary;               // does this edge locate in the boundary?
    int mTriangleId0, mTriangleId1; // The indices of the two triangles to which this side belongs. If this edge is a boundary, the mTriangleId1 is -1
};

class cTrimeshScene : public cSimScene
{
public:
    explicit cTrimeshScene();
    ~cTrimeshScene();

protected:
    tEigenArr<tTriangle *> mTriangleArray;
    tEigenArr<tEdge *> mEdgeArray;
    virtual void InitGeometry() override final;
    virtual void UpdateSubstep() override final;
    virtual void CalcTriangleDrawBuffer() override final;
    virtual void CalcEdgesDrawBuffer() override final;
};