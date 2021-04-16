#ifdef _WIN32
#include "LinctexScene.h"
#include "utils/LogUtil.h"
#include "SeScene.h"
#include "SePhysicalProperties.h"
#include "SeScene.h"
#include "SePiece.h"
#include "geometries/Primitives.h"
#include "Core/SeLogger.h"
#include "SeSimParameters.h"
#include <iostream>

SE_USING_NAMESPACE
cLinctexScene::cLinctexScene()
{
    // auto phyProp = SePhysicalProperties::Create();
    // piece = SePiece::Create(indices, pos3D, pos2D, phyProp);
    mSeScene = SeScene::Create();
    mDragPt = nullptr;
    // sim_conf = mSeScene->GetSimulationParameters();
    // SIM_INFO("init linctex succ");
    // std::cout << mSeScene->GetID() << std::endl;
    // exit(0);
}

cLinctexScene::~cLinctexScene()
{
}

void cLinctexScene::UpdateSubstep()
{
    SIM_ERROR("should not be called");
}

void cLinctexScene::InitConstraint(const Json::Value &value)
{
    cSimScene::InitConstraint(value);
}

#include "utils/JsonUtil.h"
#include "utils/MathUtil.h"
extern const tVector gGravity;

void logging(const char *a, const char *b, int c, SeLogger::Level d, const char *e)
{
    std::string prefix = "";
    switch (d)
    {
    case SeLogger::Level::Debug:
        prefix = "[debug] ";
        break;
    case SeLogger::Level::Info:
        prefix = "[info] ";
        break;
    case SeLogger::Level::Assert:
        prefix = "[assert] ";
        break;
    case SeLogger::Level::Error:
        prefix = "[error] ";
        break;
    case SeLogger::Level::Warning:
        prefix = "[warning] ";
        break;
    default:
        break;
    }
    printf("%s %s:%s : %s\n", prefix.c_str(), a, b, e);
    // std::cout << "[debug] a = " << a << std::endl;
    // std::cout << "[debug] b = " << b << std::endl;
    // std::cout << "[debug] c = " << c << std::endl;
    // std::cout << "[debug] d = " << d << std::endl;
    // std::cout << "[debug] e = " << e << std::endl;
}

void cLinctexScene::Init(const std::string &path)
{
    SeLogger::GetInstance()->RegisterCallback(logging);
    cSimScene::Init(path);
    Json::Value root;
    cJsonUtil::LoadJson(path, root);
    {
        mClothProp.Init(root);
    }

    InitGeometry(root);
    InitConstraint(root);

    // std::cout << "init cons done\n";
    // for (auto &x : mFixedPointIds)
    //     std::cout << x << std::endl;
    // exit(0);
    // add this piece to the simulator
    AddPiece();

    // init other scene
    auto ptr = mSeScene->GetSimulationParameters();
    ptr->SetGravity(gGravity[1]);
    mSeScene->Start();
    // {
    //     TriangleBaryCoord baryCoord(0.5, 0.5, 0.5);
    //     // baryCoord.coord
    //     // ;
    //     Float3 position = Float3(0.5, 0.5, 0);

    //     mCloth->AddDraggedPoint(baryCoord, position);
    // }
}

void cLinctexScene::Update(double dt)
{
    // std::cout << "update " << dt << std::endl;
    // SIM_ERROR("update hasn't been supported");
    // mSeScene->
    UpdatePerturb();
    if (mSeScene->Capture())
    {
        // std::cout << "capture\n";
        mSeScene->AcquirePositions();
        auto &pos = mCloth->FetchPositions();
        for (int i = 0; i < mVertexArray.size(); i++)
        {
            mVertexArray[i]->mPos.noalias() =
                tVector(
                    pos[i][0],
                    pos[i][1],
                    pos[i][2], 1);
        }
    }

    // else
    // {
    //     std::cout << "doesn't capture\n";
    //     exit(0);
    // }
    CalcEdgesDrawBuffer();
    CalcTriangleDrawBuffer();
}

void cLinctexScene::AddPiece()
{
    /*
SePiecePtr SePiece::Create(const std::vector<Int3> & triangles,
                    const std::vector<Float3> & positions,
                    const std::vector<Float2> & materialCoords,
                    SePhysicalPropertiesPtr pPhysicalProperties)
    */

    std::vector<Int3> indices(0);
    std::vector<Float3> pos3D(0);
    std::vector<Float2> pos2D(0);
    for (int i = 0; i < mVertexArray.size(); i++)
    {
        auto v = mVertexArray[i];
        pos3D.push_back(Float3(v->mPos[0], v->mPos[1], v->mPos[2]));
        pos2D.push_back(Float2(v->mPos[0], v->mPos[1]));
    }

    for (int i = 0; i < mTriangleArray.size(); i++)
    {
        auto tri = mTriangleArray[i];
        indices.push_back(Int3(tri->mId0, tri->mId1, tri->mId2));
        // triangle_array_se[i] = ;
        // printf("[debug] triangle %d: vertices: %d, %d, %d\n", i, tri->mId0, tri->mId1, tri->mId2);
    }

    auto phyProp = SePhysicalProperties::Create();
    phyProp->SetStretchWarp(mClothProp.mStretchWarp);
    phyProp->SetStretchWeft(mClothProp.mStretchWeft);
    phyProp->SetBendingWarp(mClothProp.mBendingWarp);
    phyProp->SetBendingWeft(mClothProp.mBendingWeft);
    std::cout << "mass density = " << phyProp->GetMassDensity() << std::endl;
    mCloth = SePiece::Create(indices, pos3D, pos2D, phyProp);

    mSeScene->AddPiece(mCloth);
    mCloth->AddFixedVertices(mFixedPointIds);
}

void cLinctexScene::ReadVertexPosFromEngine()
{
}

#include "Perturb.h"
bool cLinctexScene::CreatePerturb(tRay *ray)
{
    bool succ = cSimScene::CreatePerturb(ray);
    if (succ)
    { // 1. calculate bary
        TriangleBaryCoord bary;
        bary.coord = Vec3f(0.5, 0.5, 0.5);
        bary.index = mPerturb->mAffectedTriId;
        tVector tar_pos = mPerturb->CalcPerturbPos() + mPerturb->GetPerturbForce() / 10;
        mDragPt = mCloth->AddDraggedPoint(bary, Float3(
                                                    tar_pos[0],
                                                    tar_pos[1],
                                                    tar_pos[2]));
        // 2. caclulate triangle id
        // 3. calculate target pos
        // 4. save the perturb}
    }
    return succ;
}
void cLinctexScene::ReleasePerturb()
{
    // deconstruct
    cSimScene::ReleasePerturb();
    // unplay
    if (mDragPt != nullptr)
    {
        mCloth->Remove(mDragPt);
        mDragPt = nullptr;
    }
}
#include <SeFeatureVertices.h>
void cLinctexScene::UpdatePerturb()
{
    if (this->mDragPt != nullptr)
    {
        tVector target_pos = mPerturb->CalcPerturbPos() + mPerturb->GetPerturbForce() / 10;

        mDragPt->SetPositions(
            {Float3(
                target_pos[0],
                target_pos[1],
                target_pos[2])});
    }
}

void cLinctexScene::tPhyProperty::Init(const Json::Value &root)
{
    Json::Value conf = cJsonUtil::ParseAsValue("cloth_property", root);
    mStretchWarp = cJsonUtil::ParseAsDouble("stretch_warp", conf);
    mStretchWeft = cJsonUtil::ParseAsDouble("stretch_weft", conf);
    mBendingWarp = cJsonUtil::ParseAsDouble("bending_warp", conf);
    mBendingWeft = cJsonUtil::ParseAsDouble("bending_weft", conf);
}
#endif
