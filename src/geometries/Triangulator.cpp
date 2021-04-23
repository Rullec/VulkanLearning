#include "Triangulator.h"
#include "utils/JsonUtil.h"
#include <iostream>
void cTriangulator::BuildGeometry(const Json::Value &config, std::vector<tVertex *> &vertices_array,
                                  std::vector<tEdge *> &edges_array,
                                  std::vector<tTriangle *> &triangles_array)
{
    std::string geo_type = cJsonUtil::ParseAsString(cTriangulator::GEOMETRY_TYPE_KEY, config);
    double width = cJsonUtil::ParseAsDouble("cloth_size", config);
    double mass = cJsonUtil::ParseAsDouble("cloth_mass", config);
    tVector cloth_init_pos = tVector::Zero();
    tVector cloth_init_orientation = tVector::Zero();

    {
        cJsonUtil::ReadVectorJson(cJsonUtil::ParseAsValue("cloth_init_pos", config), cloth_init_pos);
        cJsonUtil::ReadVectorJson(cJsonUtil::ParseAsValue("cloth_init_orientation", config), cloth_init_orientation);
    }
    // std::cout << "cloth init pos = " << cloth_init_pos.transpose() << std::endl;
    // std::cout << "cloth init orientation = " << cloth_init_orientation.transpose() << std::endl;
    tMatrix init_trans_mat = cMathUtil::TransformMat(cloth_init_pos, cloth_init_orientation);
    // std::cout << init_trans_mat << std::endl;
    // exit(0);
    int subdivision = cJsonUtil::ParseAsInt("subdivision", config);
    if (geo_type == "uniform_square")
    {
        SIM_ERROR("geo type uniform_square has been deprecated, because it doesn't support bending");
        exit(0);
        cTriangulator::BuildGeometry_UniformSquare(
            width, subdivision,
            vertices_array, edges_array, triangles_array);
    }
    else if (geo_type == "skew_triangle")
    {
        cTriangulator::BuildGeometry_SkewTriangle(
            width, subdivision,
            vertices_array, edges_array, triangles_array);
    }
    else if (geo_type == "regular_triangle")
    {
        cTriangulator::BuildGeometry_UniformTriangle(
            width, subdivision,
            vertices_array, edges_array, triangles_array);
    }
    else
    {
        SIM_ERROR("unsupported geo type {}", geo_type);
    }
    ValidateGeometry(vertices_array, edges_array, triangles_array);
    // support vertices
    for (auto &v : vertices_array)
    {
        v->mMass = mass / vertices_array.size();
        v->mPos = init_trans_mat * v->mPos;
        // v->mPos.segment(0, 3) += cloth_init_pos.segment(0, 3);
    }

    // printf(
    //     "[debug] init geometry type %s, create %d vertices, %d edges, %d triangles\n", geo_type.c_str(), vertices_array.size(), edges_array.size(), triangles_array.size());
    // exit(0);
}

/**
 * \brief               Given geometry parameter, create a uniform mesh which is consist of small squares
*/
void cTriangulator::BuildGeometry_UniformSquare(double width, int subdivision, std::vector<tVertex *> &vertices_array,
                                                std::vector<tEdge *> &edges_array,
                                                std::vector<tTriangle *> &triangles_array)
{
    // 1. clear all
    vertices_array.clear();
    edges_array.clear();
    triangles_array.clear();
    // 2. create vertices

    BuildSquareVertices(width, subdivision, vertices_array);
    int num_of_lines = (subdivision + 1); // = 3
    int num_of_vertices = num_of_lines * num_of_lines;
    SIM_ASSERT(num_of_vertices = vertices_array.size());
    int num_of_edges = num_of_lines * subdivision * 2 + subdivision * subdivision;
    int num_of_triangles = num_of_lines * num_of_lines * 2;
    // 3. create edges
    double unit_edge_length = width / subdivision;
    // double unit_mass = mClothMass / (num_of_lines * num_of_lines);
    // 3. init the edges
    {
        int num_of_edges_per_line = subdivision * 3 + 1;
        int num_of_vertices_per_line = num_of_lines;
        int num_of_triangles_per_line = 2 * subdivision;
        for (int row_id = 0; row_id < subdivision + 1; row_id++)
        {
            // 3.1 add top line
            for (int col_id = 0; col_id < subdivision; col_id++)
            {
                // edge id: num_of_edges_per_line * row_id + col_id
                tEdge *edge = new tEdge();
                edge->mId0 = num_of_vertices_per_line * row_id + col_id;
                edge->mId1 = edge->mId0 + 1;
                edge->mRawLength = unit_edge_length;
                if (row_id == 0)
                {
                    edge->mIsBoundary = true;
                    edge->mTriangleId0 = 2 * col_id + 1;
                }
                else if (row_id == subdivision)
                {
                    edge->mIsBoundary = true;
                    edge->mTriangleId0 = num_of_triangles_per_line * (subdivision - 1) + col_id * 2;
                }
                else
                {
                    edge->mIsBoundary = false;
                    edge->mTriangleId0 = num_of_triangles_per_line * (row_id - 1) + 2 * col_id;
                    edge->mTriangleId1 = num_of_triangles_per_line * row_id + 2 * col_id + 1;
                }
                edges_array.push_back(edge);
                // printf("[debug] edge %d v0 %d v1 %d, is boundary %d, triangle0 %d, triangle1 %d\n",
                //    edges_array.size() - 1, edge->mId0, edge->mId1, edge->mIsBoundary, edge->mTriangleId0, edge->mTriangleId1);
            }
            if (row_id == subdivision)
                break;
            // 3.2 add middle lines

            for (int col_counting_id = 0; col_counting_id < 2 * subdivision + 1; col_counting_id++)
            {
                int col_id = col_counting_id / 2;
                tEdge *edge = new tEdge();
                if (col_counting_id % 2 == 0)
                {
                    // vertical line
                    edge->mId0 = row_id * num_of_vertices_per_line + col_id;
                    edge->mId1 = (row_id + 1) * num_of_vertices_per_line + col_id;
                    edge->mRawLength = unit_edge_length;
                    if (col_id == 0)
                    {
                        // left edge
                        edge->mIsBoundary = true;
                        edge->mTriangleId0 = num_of_triangles_per_line * row_id;
                    }
                    else if (col_counting_id == 2 * subdivision)
                    {
                        // right edge
                        edge->mIsBoundary = true;
                        edge->mTriangleId0 = num_of_triangles_per_line * (row_id + 1) - 1;
                    }
                    else
                    {
                        // middle edges
                        edge->mIsBoundary = false;
                        edge->mTriangleId0 = num_of_triangles_per_line * row_id + col_id;
                        edge->mTriangleId1 = num_of_triangles_per_line * row_id + col_id + 1;
                    }
                }
                else
                {
                    continue;
                    std::cout << "ignore skew edge\n";
                    // skew line
                    // edge->mId0 = num_of_vertices_per_line * row_id + col_id;
                    // edge->mId1 = num_of_vertices_per_line * (row_id + 1) + col_id + 1;
                    // edge->mIsBoundary = false;
                    // edge->mRawLength = unit_edge_length * std::sqrt(2);
                    // edge->mTriangleId0 = num_of_triangles * row_id + col_id * 2;
                    // edge->mTriangleId1 = edge->mTriangleId0 + 1;
                }
                edges_array.push_back(edge);
                // printf("[debug] edge %d v0 %d v1 %d, is boundary %d, triangle0 %d, triangle1 %d\n",
                //    edges_array.size() - 1, edge->mId0, edge->mId1, edge->mIsBoundary, edge->mTriangleId0, edge->mTriangleId1);
            }
        }
    }
}

/**
 * \brief           Create skew triangles
*/
void cTriangulator::BuildGeometry_SkewTriangle(double width, int subdivision, std::vector<tVertex *> &vertices_array,
                                               std::vector<tEdge *> &edges_array,
                                               std::vector<tTriangle *> &triangles_array)
{
    // 1. clear all
    vertices_array.clear();
    edges_array.clear();
    triangles_array.clear();

    // 2. create vertices
    BuildSquareVertices(width, subdivision, vertices_array);

    // 3. create triangles
    int num_of_lines = (subdivision + 1); // = 3
    int num_of_vertices = num_of_lines * num_of_lines;
    int num_of_edges = num_of_lines * subdivision * 2 + subdivision * subdivision;
    int num_of_triangles = num_of_lines * num_of_lines * 2;

    // 1. init the triangles
    for (int row_id = 0; row_id < subdivision; row_id++)
    {
        for (int col_id = 0; col_id < subdivision; col_id++)
        {
            int up_left = row_id * num_of_lines + col_id;
            auto tri1 = new tTriangle(up_left, up_left + num_of_lines, up_left + 1 + num_of_lines);
            auto tri2 = new tTriangle(up_left, up_left + 1 + num_of_lines, up_left + 1);

            triangles_array.push_back(tri1);
            // printf("[debug] triangle %d vertices %d %d %d\n", triangles_array.size() - 1, tri1->mId0, tri1->mId1, tri1->mId2);
            triangles_array.push_back(tri2);
            // printf("[debug] triangle %d vertices %d %d %d\n", triangles_array.size() - 1, tri2->mId0, tri2->mId1, tri2->mId2);
        }
    }
    printf("[debug] create %d triangles done\n", triangles_array.size());
    // 2. init the vertices
    double unit_edge_length = width / subdivision;
    // double unit_mass = mClothMass / (num_of_lines * num_of_lines);

    printf("[debug] create %d vertices done\n", vertices_array.size());
    // 3. init the edges
    {
        int num_of_edges_per_line = subdivision * 3 + 1;
        int num_of_vertices_per_line = num_of_lines;
        int num_of_triangles_per_line = 2 * subdivision;
        for (int row_id = 0; row_id < subdivision + 1; row_id++)
        {
            // 3.1 add top line
            for (int col_id = 0; col_id < subdivision; col_id++)
            {
                // edge id: num_of_edges_per_line * row_id + col_id
                tEdge *edge = new tEdge();
                edge->mId0 = num_of_vertices_per_line * row_id + col_id;
                edge->mId1 = edge->mId0 + 1;
                edge->mRawLength = unit_edge_length;
                if (row_id == 0)
                {
                    edge->mIsBoundary = true;
                    edge->mTriangleId0 = 2 * col_id + 1;
                }
                else if (row_id == subdivision)
                {
                    edge->mIsBoundary = true;
                    edge->mTriangleId0 = num_of_triangles_per_line * (subdivision - 1) + col_id * 2;
                }
                else
                {
                    edge->mIsBoundary = false;
                    edge->mTriangleId0 = num_of_triangles_per_line * (row_id - 1) + 2 * col_id;
                    edge->mTriangleId1 = num_of_triangles_per_line * row_id + 2 * col_id + 1;
                }
                edges_array.push_back(edge);
                // printf("[debug] edge %d v0 %d v1 %d, is boundary %d, triangle0 %d, triangle1 %d\n",
                //    edges_array.size() - 1, edge->mId0, edge->mId1, edge->mIsBoundary, edge->mTriangleId0, edge->mTriangleId1);
            }
            if (row_id == subdivision)
                break;
            // 3.2 add middle lines

            for (int col_counting_id = 0; col_counting_id < 2 * subdivision + 1; col_counting_id++)
            {
                int col_id = col_counting_id / 2;
                tEdge *edge = new tEdge();
                if (col_counting_id % 2 == 0)
                {
                    // vertical line
                    edge->mId0 = row_id * num_of_vertices_per_line + col_id;
                    edge->mId1 = (row_id + 1) * num_of_vertices_per_line + col_id;
                    edge->mRawLength = unit_edge_length;
                    if (col_id == 0)
                    {
                        // left edge
                        edge->mIsBoundary = true;
                        edge->mTriangleId0 = num_of_triangles_per_line * row_id;
                    }
                    else if (col_counting_id == 2 * subdivision)
                    {
                        // right edge
                        edge->mIsBoundary = true;
                        edge->mTriangleId0 = num_of_triangles_per_line * (row_id + 1) - 1;
                    }
                    else
                    {
                        // middle edges
                        edge->mIsBoundary = false;
                        edge->mTriangleId0 = num_of_triangles_per_line * row_id + col_id;
                        edge->mTriangleId1 = num_of_triangles_per_line * row_id + col_id + 1;
                    }
                }
                else
                {
                    // continue;
                    // std::cout << "ignore skew edge\n";
                    // skew line
                    edge->mId0 = num_of_vertices_per_line * row_id + col_id;
                    edge->mId1 = num_of_vertices_per_line * (row_id + 1) + col_id + 1;
                    edge->mIsBoundary = false;
                    edge->mRawLength = unit_edge_length * std::sqrt(2);
                    edge->mTriangleId0 = num_of_triangles_per_line * row_id + col_id * 2;
                    edge->mTriangleId1 = edge->mTriangleId0 + 1;
                }
                edges_array.push_back(edge);
                // printf("[debug] edge %d v0 %d v1 %d, is boundary %d, triangle0 %d, triangle1 %d\n",
                //    edges_array.size() - 1, edge->mId0, edge->mId1, edge->mIsBoundary, edge->mTriangleId0, edge->mTriangleId1);
            }
        }
    }
    printf("[debug] create %d edges done\n", edges_array.size());
}
void cTriangulator::BuildGeometry_UniformTriangle(double width, int subdivision, std::vector<tVertex *> &vertices_array,
                                                  std::vector<tEdge *> &edges_array,
                                                  std::vector<tTriangle *> &triangles_array)
{
    // 1. clear all
    vertices_array.clear();
    edges_array.clear();
    triangles_array.clear();

    // 2. create vertices
    BuildSquareVertices(width, subdivision, vertices_array);

    // 3. create triangles
    int num_of_lines = (subdivision + 1); // = 3
    int num_of_vertices = num_of_lines * num_of_lines;
    int num_of_edges = num_of_lines * subdivision * 2 + subdivision * subdivision;
    int num_of_triangles = num_of_lines * num_of_lines * 2;

    // 1. init the triangles
    for (int row_id = 0; row_id < subdivision; row_id++)
    {
        for (int col_id = 0; col_id < subdivision; col_id++)
        {
            // for even number, from upleft to downright
            int up_left = row_id * num_of_lines + col_id;
            if (col_id % 2 == 0)
            {

                auto tri1 = new tTriangle(up_left, up_left + num_of_lines, up_left + 1 + num_of_lines);
                auto tri2 = new tTriangle(up_left, up_left + 1 + num_of_lines, up_left + 1);

                triangles_array.push_back(tri1);
                // printf("[debug] triangle %d vertices %d %d %d\n", triangles_array.size() - 1, tri1->mId0, tri1->mId1, tri1->mId2);
                triangles_array.push_back(tri2);
                // printf("[debug] triangle %d vertices %d %d %d\n", triangles_array.size() - 1, tri2->mId0, tri2->mId1, tri2->mId2);
            }
            else
            {
                // for odd number, from upright to downleft
                auto tri1 = new tTriangle(up_left + 1, up_left, up_left + num_of_lines);
                auto tri2 = new tTriangle(up_left + 1, up_left + num_of_lines, up_left + 1 + num_of_lines);

                triangles_array.push_back(tri1);
                // printf("[debug] triangle %d vertices %d %d %d\n", triangles_array.size() - 1, tri1->mId0, tri1->mId1, tri1->mId2);
                triangles_array.push_back(tri2);
                // printf("[debug] triangle %d vertices %d %d %d\n", triangles_array.size() - 1, tri2->mId0, tri2->mId1, tri2->mId2);
            }
        }
    }
    printf("[debug] create %d triangles done\n", triangles_array.size());
    // 2. init the vertices
    double unit_edge_length = width / subdivision;
    // double unit_mass = mClothMass / (num_of_lines * num_of_lines);

    printf("[debug] create %d vertices done\n", vertices_array.size());
    // 3. init the edges
    {
        int num_of_edges_per_line = subdivision * 3 + 1;
        int num_of_vertices_per_line = num_of_lines;
        int num_of_triangles_per_line = 2 * subdivision;
        for (int row_id = 0; row_id < subdivision + 1; row_id++)
        {
            // 3.1 add top line
            for (int col_id = 0; col_id < subdivision; col_id++)
            {
                // edge id: num_of_edges_per_line * row_id + col_id
                tEdge *edge = new tEdge();
                edge->mId0 = num_of_vertices_per_line * row_id + col_id;
                edge->mId1 = edge->mId0 + 1;
                edge->mRawLength = unit_edge_length;
                if (row_id == 0)
                {
                    edge->mIsBoundary = true;
                    if (col_id % 2 == 0)
                    {

                        edge->mTriangleId0 = 2 * col_id + 1;
                    }
                    else
                    {
                        edge->mTriangleId0 = 2 * col_id;
                    }
                }
                else if (row_id == subdivision)
                {
                    edge->mIsBoundary = true;

                    if (col_id % 2 == 0)
                    {

                        edge->mTriangleId0 = num_of_triangles_per_line * (subdivision - 1) + col_id * 2;
                    }
                    else
                    {
                        edge->mTriangleId0 = num_of_triangles_per_line * (subdivision - 1) + col_id * 2 + 1;
                    }
                }
                else
                {
                    edge->mIsBoundary = false;
                    if (col_id % 2 == 0)
                    {

                        edge->mTriangleId0 = num_of_triangles_per_line * (row_id - 1) + 2 * col_id;
                        edge->mTriangleId1 = num_of_triangles_per_line * row_id + 2 * col_id + 1;
                    }
                    else
                    {
                        edge->mTriangleId0 = num_of_triangles_per_line * (row_id - 1) + 2 * col_id + 1;
                        edge->mTriangleId1 = num_of_triangles_per_line * row_id + 2 * col_id;
                    }
                }
                edges_array.push_back(edge);
                // printf("[debug] edge %d v0 %d v1 %d, is boundary %d, triangle0 %d, triangle1 %d\n",
                //        edges_array.size() - 1, edge->mId0, edge->mId1, edge->mIsBoundary, edge->mTriangleId0, edge->mTriangleId1);
            }
            if (row_id == subdivision)
                break;
            // 3.2 add middle lines

            for (int col_counting_id = 0; col_counting_id < 2 * subdivision + 1; col_counting_id++)
            {
                int col_id = col_counting_id / 2;
                tEdge *edge = new tEdge();

                if (col_counting_id % 2 == 0)
                {
                    // vertical line
                    edge->mId0 = row_id * num_of_vertices_per_line + col_id;
                    edge->mId1 = (row_id + 1) * num_of_vertices_per_line + col_id;
                    edge->mRawLength = unit_edge_length;
                    if (col_id == 0)
                    {
                        // left edge
                        edge->mIsBoundary = true;
                        edge->mTriangleId0 = num_of_triangles_per_line * row_id;
                    }
                    else if (col_counting_id == 2 * subdivision)
                    {
                        // right edge
                        edge->mIsBoundary = true;
                        edge->mTriangleId0 = num_of_triangles_per_line * (row_id + 1) - 1;
                    }
                    else
                    {
                        // middle edges
                        edge->mIsBoundary = false;
                        edge->mTriangleId0 = num_of_triangles_per_line * row_id + col_counting_id - 1;
                        edge->mTriangleId1 = edge->mTriangleId0 + 1;
                    }
                }
                else
                {
                    // continue;
                    // std::cout << "ignore skew edge\n";
                    // skew line
                    if (col_id % 2 == 0)
                    {
                        edge->mId0 = num_of_vertices_per_line * row_id + col_id;
                        edge->mId1 = num_of_vertices_per_line * (row_id + 1) + col_id + 1;
                    }
                    else
                    {
                        edge->mId0 = num_of_vertices_per_line * row_id + col_id + 1;
                        edge->mId1 = num_of_vertices_per_line * (row_id + 1) + col_id;
                    }
                    edge->mIsBoundary = false;
                    edge->mRawLength = unit_edge_length * std::sqrt(2);
                    edge->mTriangleId0 = num_of_triangles_per_line * row_id + col_id * 2;
                    edge->mTriangleId1 = edge->mTriangleId0 + 1;
                }
                edges_array.push_back(edge);

                // printf("[debug] edge %d v0 %d v1 %d, is boundary %d, triangle0 %d, triangle1 %d\n",
                //        edges_array.size() - 1, edge->mId0, edge->mId1, edge->mIsBoundary, edge->mTriangleId0, edge->mTriangleId1);
            }
        }
    }
    printf("[debug] create %d edges done\n", edges_array.size());
    // exit(0);
}

/**
 * \brief                   create vertices as a uniform, square vertices
*/
void cTriangulator::BuildSquareVertices(
    double width, int subdivision,
    std::vector<tVertex *> &vertices_array)
{
    vertices_array.clear();
    int gap = subdivision + 1;
    double unit_edge_length = width / subdivision;

    for (int i = 0; i < gap; i++)
        for (int j = 0; j < gap; j++)
        {
            tVertex *v = new tVertex();

            v->mPos =
                tVector(unit_edge_length * j - width / 2, width - unit_edge_length * i - width / 2, 0, 1);
            v->mPos[3] = 1;
            v->mColor = tVector(0, 196.0 / 255, 1, 0);
            vertices_array.push_back(v);
            v->muv =
                tVector2f(i * 1.0 / subdivision, j * 1.0 / subdivision);
            // std::cout << "uv = " << v->muv.transpose() << std::endl;
            // printf("create vertex %d at (%.3f, %.3f), uv (%.3f, %.3f)\n",
            //        vertices_array.size() - 1, v->mPos[0], v->mPos[1],
            //        v->muv[0], v->muv[1]);
        }
    // exit(0);
}

bool ConfirmVertexInTriangles(tTriangle *tri, int vid)
{
    return (tri->mId0 == vid) ||
           (tri->mId1 == vid) ||
           (tri->mId2 == vid);
};
void cTriangulator::ValidateGeometry(std::vector<tVertex *> &vertices_array,
                                     std::vector<tEdge *> &edges_array,
                                     std::vector<tTriangle *> &triangles_array)
{
    // confirm the edges is really shared by triangles
    for (int i = 0; i < edges_array.size(); i++)
    {
        auto &e = edges_array[i];
        if (e->mTriangleId0 != -1)
        {
            auto tri = triangles_array[e->mTriangleId0];
            if (
                (
                    ConfirmVertexInTriangles(tri, e->mId0) &&
                    ConfirmVertexInTriangles(tri, e->mId1)) == false)
            {
                printf("[error] validate geometry adjoint edge %d failed between tri%d - tri%d\n",
                       i, e->mTriangleId0, e->mTriangleId1);
                exit(0);
            }
        }
        if (e->mTriangleId1 != -1)
        {
            auto tri = triangles_array[e->mTriangleId1];
            if (
                (
                    ConfirmVertexInTriangles(tri, e->mId0) &&
                    ConfirmVertexInTriangles(tri, e->mId1)) == false)
            {
                printf("[error] validate geometry adjoint edge %d failed between tri%d - tri%d\n",
                       i, e->mTriangleId1, e->mTriangleId1);
                exit(0);
            }
        }
    }
}

/**
 * \brief           Given the geometry info, save them to the given "path"
 * 
 *          Only save a basic info
*/
void cTriangulator::SaveGeometry(std::vector<tVertex *> &vertices_array,
                                 std::vector<tEdge *> &edges_array,
                                 std::vector<tTriangle *> &triangles_array,
                                 const std::string &path)
{
    Json::Value root;
    // 1. the vertices info
    root[NUM_OF_VERTICES_KEY] = vertices_array.size();

    // 2. the edge info

    root[EDGE_ARRAY_KEY] = Json::arrayValue;
    for (auto &x : edges_array)
    {
        root[EDGE_ARRAY_KEY].append(x->mId0);
        root[EDGE_ARRAY_KEY].append(x->mId1);
    }
    // 3. the triangle info
    root[TRIANGLE_ARRAY_KEY] = Json::arrayValue;
    for (auto &x : triangles_array)
    {
        Json::Value tri = Json::arrayValue;

        tri.append(x->mId0);
        tri.append(x->mId1);
        tri.append(x->mId2);
        root[TRIANGLE_ARRAY_KEY].append(tri);
    }
    std::cout << "[debug] save geometry to " << path << std::endl;
    cJsonUtil::WriteJson(path, root, true);
}

void cTriangulator::LoadGeometry(std::vector<tVertex *> &vertices_array,
                                 std::vector<tEdge *> &edges_array,
                                 std::vector<tTriangle *> &triangles_array,
                                 const std::string &path)
{
    vertices_array.clear();
    edges_array.clear();
    triangles_array.clear();
    Json::Value root;
    cJsonUtil::LoadJson(path, root);
    int num_of_vertices = cJsonUtil::ParseAsInt(NUM_OF_VERTICES_KEY, root);
    for (int i = 0; i < num_of_vertices; i++)
    {
        vertices_array.push_back(new tVertex());
        vertices_array[vertices_array.size() - 1]->mColor = tVector(0, 196.0 / 255, 1, 0);
    }

    const tVectorXd &edge_info =
        cJsonUtil::ReadVectorJson(cJsonUtil::ParseAsValue(EDGE_ARRAY_KEY, root));

    for (int i = 0; i < edge_info.size() / 2; i++)
    {
        tEdge *edge = new tEdge();
        edge->mId0 = edge_info[2 * i + 0];
        edge->mId1 = edge_info[2 * i + 1];
        edges_array.push_back(edge);
    }

    const Json::Value &tri_json = cJsonUtil::ParseAsValue(TRIANGLE_ARRAY_KEY, root);
    for (int i = 0; i < tri_json.size(); i++)
    {
        tTriangle *tri = new tTriangle();
        tri->mId0 = tri_json[i][0].asInt();
        tri->mId1 = tri_json[i][1].asInt();
        tri->mId2 = tri_json[i][2].asInt();
        triangles_array.push_back(tri);
    }
    printf("[debug] Load Geometry from %s done, vertices %d, edges %d, triangles %d\n",
           path.c_str(),
           vertices_array.size(),
           edges_array.size(),
           triangles_array.size());
}
