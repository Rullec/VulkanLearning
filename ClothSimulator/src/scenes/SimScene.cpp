#include "SimScene.h"
#include "utils/JsonUtil.h"
#include <iostream>

tVertex::tVertex()
{
    mMass = 0;
    mPos.setZero();
    mColor = tVector::Ones();
}

tSpring::tSpring()
{
    mRawLength = 0;
    mK = 0;
    mId0 = -1;
    mId1 = -1;
}

cSimScene::cSimScene()
{
    mVertexArray.clear();
    mSpringArray.clear();
}

void cSimScene::Init(const std::string &conf_path)
{
    // 1. load config
    Json::Value root;
    cJsonUtil::LoadJson(conf_path, root);

    mClothWidth = cJsonUtil::ParseAsDouble("cloth_size", root);
    mSubdivision = cJsonUtil::ParseAsInt("subdivision", root);
    SIM_INFO("cloth total width {} subdivision {}", mClothWidth, mSubdivision);
    // 2. create geometry, dot allocation
    InitGeometry();
    std::cout << "init sim scene done\n";
}

/**
 * \brief           Update the simulation procedure
*/
void cSimScene::Update(double dt)
{
    // 1. clear force
    ClearForce();
    // 2. calculate force
    CalcIntForce();
    CalcExtForce();
    // 3. forward simulation
    CalcNextPosition();
    // 4. post process
    CalcTriangleDrawBuffer();
    CalcEdgesDrawBuffer();
}

/**
 * \brief           Reset the whole scene
*/
void cSimScene::Reset()
{
    ClearForce();
}

/**
 * \brief           Get number of vertices
*/
int cSimScene::GetNumOfVertices() const
{
    return mVertexArray.size();
}

/**
 * \brief   discretazation from square cloth to mass spring system
 * 
*/
void cSimScene::InitGeometry()
{
    int spring_id = 0;
    int vertex_id = 0;
    int gap = mSubdivision + 1;
    double unit_edge_length = mClothWidth / mSubdivision;
    // for all row lines' edges
    for (int i = 0; i < mSubdivision + 1; i++)
    {
        for (int j = 0; j < mSubdivision; j++)
        {
            // if i is row index
            {
                tSpring *spr = new tSpring();
                spr->mRawLength = unit_edge_length;
                spr->mK = 1;
                spr->mId0 = gap * i + j;
                spr->mId1 = gap * i + j + 1;
                mSpringArray.push_back(spr);
                printf("create spring %d between %d and %d\n", mSpringArray.size() - 1, spr->mId0, spr->mId1);
            }
            // if i is column index
            {
                tSpring *spr = new tSpring();
                spr->mRawLength = unit_edge_length;
                spr->mK = 1;
                spr->mId0 = gap * j + i;
                spr->mId1 = gap * (j + 1) + i;
                mSpringArray.push_back(spr);
                printf("create spring %d between %d and %d\n", mSpringArray.size() - 1, spr->mId0, spr->mId1);
            }
        }
    }

    // set up the vertex pos data
    // in XOY plane
    {
        for (int i = 0; i < gap; i++)
            for (int j = 0; j < gap; j++)
            {
                tVertex *v = new tVertex();
                v->mMass = 1;
                v->mPos = tVector(
                    unit_edge_length * i,
                    unit_edge_length * j,
                    0,
                    1);
                v->mColor = tVector(0, 196.0 / 255, 1, 0);
                mVertexArray.push_back(v);
                printf("create vertex %d at (%.3f, %.3f)\n", mVertexArray.size() - 1, v->mPos[0], v->mPos[1]);
            }
    }

    // init the buffer
    {
        int num_of_square = mSubdivision * mSubdivision;
        int num_of_triangles = num_of_square * 2;
        int num_of_vertices = num_of_triangles * 3;
        int size_per_vertices = 8;
        mTriangleDrawBuffer.resize(num_of_vertices * size_per_vertices);
    }
    {
        int num_of_edges = 2 * (gap - 1) * gap;
        int size_per_edge = 16;
        mEdgesDrawBuffer.resize(num_of_edges * size_per_edge);
    }

    CalcTriangleDrawBuffer();
    CalcEdgesDrawBuffer();
}

/**
 * \brief       clear all forces
*/
void cSimScene::ClearForce()
{
    int dof = GetNumOfFreedom();
    mIntForce.noalias() = tVectorXd::Zero(dof);
}

/**
 * \brief            calculate inv mass mat
*/
void cSimScene::CalcInvMassMatrix() const
{
}

/**
 * \brief       external force
*/
void cSimScene::CalcExtForce() const
{
}

/**
 * \brief       internal force
*/
void cSimScene::CalcIntForce() const
{
}

/**
 * \brief       Given total force, calcualte the next vertices' position
*/
void cSimScene::CalcNextPosition()
{
}
void cSimScene::GetVertexRenderingData()
{
}

int cSimScene::GetNumOfFreedom() const
{
    return GetNumOfVertices() * 3;
}

cSimScene::~cSimScene()
{
    for (auto x : mVertexArray)
        delete x;
    for (auto x : mSpringArray)
        delete x;

    mVertexArray.clear();
    mSpringArray.clear();
}
void CalcTriangleDrawBufferSingle(tVertex *v0, tVertex *v1, tVertex *v2, tVectorXf &buffer, int &st_pos)
{
    // std::cout << "buffer size " << buffer.size() << " st pos " << st_pos << std::endl;
    buffer.segment(st_pos, 3) = v0->mPos.segment(0, 3).cast<float>();
    buffer.segment(st_pos + 3, 3) = v0->mColor.segment(0, 3).cast<float>();
    st_pos += 8;
    buffer.segment(st_pos, 3) = v1->mPos.segment(0, 3).cast<float>();
    buffer.segment(st_pos + 3, 3) = v1->mColor.segment(0, 3).cast<float>();
    st_pos += 8;
    buffer.segment(st_pos, 3) = v2->mPos.segment(0, 3).cast<float>();
    buffer.segment(st_pos + 3, 3) = v2->mColor.segment(0, 3).cast<float>();
    st_pos += 8;
}
void CalcEdgeDrawBufferSingle(tVertex *v0, tVertex *v1, tVectorXf &buffer, int &st_pos)
{

    buffer.segment(st_pos, 3) = v0->mPos.segment(0, 3).cast<float>();
    buffer.segment(st_pos + 3, 3) = tVector3f(0, 0, 0);
    st_pos += 8;
    buffer.segment(st_pos, 3) = v1->mPos.segment(0, 3).cast<float>();
    buffer.segment(st_pos + 3, 3) = tVector3f(0, 0, 0);
    st_pos += 8;
}

const tVectorXf &cSimScene::GetTriangleDrawBuffer()
{
    return mTriangleDrawBuffer;
}
/**
 * \brief           Calculate vertex rendering data
*/
void cSimScene::CalcTriangleDrawBuffer()
{
    mTriangleDrawBuffer.fill(std::nan(""));
    // counter clockwise
    int gap = mSubdivision + 1;
    int st = 0;
    for (int i = 0; i < mSubdivision; i++)     // row
        for (int j = 0; j < mSubdivision; j++) // column
        {
            // left up coner
            int left_up = gap * i + j;
            int right_up = left_up + 1;
            int left_down = left_up + gap;
            int right_down = right_up + gap;
            // mVertexArray[left_up]->mPos *= (1 + 1e-3);
            CalcTriangleDrawBufferSingle(mVertexArray[right_down], mVertexArray[left_up], mVertexArray[left_down], mTriangleDrawBuffer, st);
            CalcTriangleDrawBufferSingle(mVertexArray[right_down], mVertexArray[right_up], mVertexArray[left_up], mTriangleDrawBuffer, st);
        }
}

const tVectorXf &cSimScene::GetEdgesDrawBuffer()
{
    return mEdgesDrawBuffer;
}

void cSimScene::CalcEdgesDrawBuffer()
{
    mEdgesDrawBuffer.fill(std::nan(""));
    int st = 0;
    int gap = mSubdivision + 1;

    // for all row lines' edges
    for (int i = 0; i < mSubdivision + 1; i++)
    {
        for (int j = 0; j < mSubdivision; j++)
        {
            // printf("[debug] edge from %d to %d: st %d / %d\n", i, j, st, mEdgesDrawBuffer.size());
            // if i is row index
            {
                int Id0 = gap * i + j;
                int Id1 = gap * i + j + 1;
                CalcEdgeDrawBufferSingle(mVertexArray[Id0], mVertexArray[Id1], mEdgesDrawBuffer, st);
            }
            // if i is column index
            {

                int Id0 = gap * j + i;
                int Id1 = gap * (j + 1) + i;
                CalcEdgeDrawBufferSingle(mVertexArray[Id0], mVertexArray[Id1], mEdgesDrawBuffer, st);
            }
        }
    }
}