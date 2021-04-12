#pragma once
#include "Primitives.h"
namespace Json
{
    class Value;
};
class cTriangulator
{
public:
    static void BuildGeometry(const Json::Value &config, std::vector<tVertex *> &vertices_array,
                              std::vector<tEdge *> &edges_array,
                              std::vector<tTriangle *> &triangles_array);

protected:
    static void BuildGeometry_UniformSquare(double width, int subdivistion, std::vector<tVertex *> &vertices_array,
                                            std::vector<tEdge *> &edges_array,
                                            std::vector<tTriangle *> &triangles_array);
    static void BuildGeometry_SkewTriangle(double width, int subdivistion, std::vector<tVertex *> &vertices_array,
                                           std::vector<tEdge *> &edges_array,
                                           std::vector<tTriangle *> &triangles_array);
    static void BuildGeometry_UniformTriangle(double width, int subdivistion, std::vector<tVertex *> &vertices_array,
                                              std::vector<tEdge *> &edges_array,
                                              std::vector<tTriangle *> &triangles_array);
    static void BuildSquareVertices(
        double width, int subdivision,
        std::vector<tVertex *> &edges_array);
    static void ValidateGeometry(std::vector<tVertex *> &vertices_array,
                                 std::vector<tEdge *> &edges_array,
                                 std::vector<tTriangle *> &triangles_array);
};