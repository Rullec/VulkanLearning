#pragma once
#include <string>
#include <vector>

struct tTriangle;
struct tEdge;
struct tVertex;

class cObjUtil
{
public:
    struct tParams
    {
        std::string mPath; // obj file path
    };

    static void LoadObj(const tParams &param,
                        std::vector<tVertex *> &mVertexArray,
                        std::vector<tEdge *> &mEdgeArray,
                        std::vector<tTriangle *> &mTriangleArray);

protected:
    static void BuildEdge(const std::vector<tVertex *> &mVertexArray,
                          std::vector<tEdge *> &mEdgeArray,
                          const std::vector<tTriangle *> &mTriangleArray);
};