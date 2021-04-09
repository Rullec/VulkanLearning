#include "Primitives.h"
tVertex::tVertex()
{
    mMass = 0;
    mPos = tVector(0, 0, 0, 1);
    muv.setZero();
    mColor.setZero();
}

tEdge::tEdge()
{
    mId0 = mId1 = -1;
    mRawLength = 0;
    mIsBoundary = false;
    mTriangleId0 = mTriangleId1 = -1;
    mK_spring = 0;
}

tTriangle::tTriangle()
{
    mId0 = mId1 = mId2 = -1;
}
tTriangle::tTriangle(int a, int b, int c) : mId0(a), mId1(b), mId2(c)

{
}
