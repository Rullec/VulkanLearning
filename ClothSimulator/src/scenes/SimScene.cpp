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
    CalcVertexRenderingData();
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
                v->mColor = tVector::Ones();
                mVertexArray.push_back(v);
                printf("create vertex %d at (%.3f, %.3f)\n", mVertexArray.size() - 1, v->mPos[0], v->mPos[1]);
            }
    }
    // exit(0);
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

/**
 * \brief           Calculate vertex rendering data
*/
void cSimScene::CalcVertexRenderingData() //
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