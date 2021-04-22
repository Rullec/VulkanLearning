#ifdef _WIN32
#include "LinctexScene.h"
#include "utils/LogUtil.h"
#include "utils/FileUtil.h"
#include "SeScene.h"
#include "SePhysicalProperties.h"
#include "SeSimulationProperties.h"
#include "SeScene.h"
#include "SePiece.h"
#include "geometries/Primitives.h"
#include "Core/SeLogger.h"
#include "SeSimParameters.h"
#include "SeSceneOptions.h"
#include "sim/ClothProperty.h"
#include <iostream>
#include <thread> // std::this_thread::sleep_for
#include <chrono> // std::chrono::seconds
SE_USING_NAMESPACE
cLinctexScene::cLinctexScene()
{
    // auto phyProp = SePhysicalProperties::Create();
    // piece = SePiece::Create(indices, pos3D, pos2D, phyProp);
    mSeScene = SeScene::Create();
    mSeScene->GetOptions()->SetPlatForm(SePlatform::CUDA);
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
    if (d != SeLogger::Level::Error)
        return;
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
void cLinctexScene::Reset()
{
    mSeScene->End();
    mEngineStart = false;
    UpdateCurNodalPosition(mClothInitPos);
    UpdateClothFeatureVector();
}
void cLinctexScene::Init(const std::string &path)
{
    SeLogger::GetInstance()->RegisterCallback(logging);
    cSimScene::Init(path);
    Json::Value root;
    cJsonUtil::LoadJson(path, root);
    {
        mClothProp = std::make_shared<tPhyProperty>();
        mClothProp->Init(root);
    }
    {
        mEnableNetworkInferenceMode = cJsonUtil::ParseAsBool("enable_network_inference_mode", root);
        if (mEnableNetworkInferenceMode == true)
        {
            mNetworkInfer_ConvThreshold = 0;
            mNetworkInfer_OutputPath = "";
            mNetworkInfer_ConvThreshold = cJsonUtil::ParseAsDouble("network_inference_convergence_threshold", root);
            mNetworkInfer_OutputPath = cJsonUtil::ParseAsString("network_inference_output_path", root);
            mNetworkInfer_MinIter = cJsonUtil::ParseAsInt("network_inference_min_iter", root);
            mPreviosFeature.resize(0);
            mNetworkInfer_CurIter = 0;
            std::cout << "[NN] output path = " << mNetworkInfer_OutputPath << std::endl;
            std::cout << "[NN] conv threshold = " << mNetworkInfer_ConvThreshold << std::endl;
            // exit(0);
        }
    }

    InitGeometry(root);
    InitConstraint(root);
    InitDrawBuffer();
    InitClothFeatureVector();
    // std::cout << "init cons done\n";
    // for (auto &x : mFixedPointIds)
    //     std::cout << x << std::endl;
    // exit(0);
    // add this piece to the simulator
    AddPiece();

    // init other scene
    auto ptr = mSeScene->GetSimulationParameters();
    ptr->SetGravity(gGravity[1]);
    mEngineStart = false;
    // ptr->GetCollisionThickness();
    auto cloth_sim_prop = mCloth->GetSimulationProperties();
    // cloth_sim_prop->SetCollisionThickness(0);
    mCloth->GetPhysicalProperties()->SetThickness(0);
    std::cout << "[debug] se col thickness = " << cloth_sim_prop->GetCollisionThickness() << std::endl;
    // exit(0);
    // mSeScene->GetOptions(
    // {
    //     TriangleBaryCoord baryCoord(0.5, 0.5, 0.5);
    //     // baryCoord.coord
    //     // ;
    //     Float3 position = Float3(0.5, 0.5, 0);

    //     mCloth->AddDraggedPoint(baryCoord, position);
    // }
}
void cLinctexScene::NetworkInferenceFunction()
{
    bool convergence = true;
    if (mPreviosFeature.size() != GetNumOfFreedom())
    {
        // the first time
        convergence = false;
    }
    else
    {
        double cur_norm = (mPreviosFeature - mClothFeature).norm();
        std::cout << "iter " << mNetworkInfer_CurIter << " cur norm = " << cur_norm << std::endl;
        if (cur_norm > mNetworkInfer_ConvThreshold || mNetworkInfer_CurIter < mNetworkInfer_MinIter)
        {
            convergence = false;
        }
    }
    mPreviosFeature = mClothFeature;
    if (convergence)
    {
        std::cout << "exit, save current result to " << mNetworkInfer_OutputPath << std::endl;
        cLinctexScene::DumpSimulationData(
            mClothFeature,
            mClothProp->BuildFeatureVector(),
            tVector::Zero(),
            tVector::Zero(),
            mNetworkInfer_OutputPath);
        exit(0);
    }
    mNetworkInfer_CurIter++;
}
void cLinctexScene::Update(double dt)
{
    // std::cout << "linctex update\n";
    if (mEngineStart == false)
    {
        mEngineStart = true;
        mSeScene->Start();
    }
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
        UpdateClothFeatureVector();

        if (mEnableNetworkInferenceMode == true)
            NetworkInferenceFunction();
    }
    // std::cout << "[debug] cloth feature norm = " << GetClothFeatureVector().norm() << std::endl;
    // else
    // {
    //     std::cout << "doesn't capture\n";
    //     exit(0);
    // }
    // CalcEdgesDrawBuffer();
    // CalcTriangleDrawBuffer();
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
        pos2D.push_back(Float2(v->muv[0], v->muv[1]) * mClothWidth);
    }

    for (int i = 0; i < mTriangleArray.size(); i++)
    {
        auto tri = mTriangleArray[i];
        indices.push_back(Int3(tri->mId0, tri->mId1, tri->mId2));
        // triangle_array_se[i] = ;
        // printf("[debug] triangle %d: vertices: %d, %d, %d\n", i, tri->mId0, tri->mId1, tri->mId2);
    }

    auto phyProp = SePhysicalProperties::Create();
    // std::cout << "mass density = " << phyProp->GetMassDensity() << std::endl;
    mCloth = SePiece::Create(indices, pos3D, pos2D, phyProp);
    SetSimProperty(mClothProp);
    mSeScene->AddPiece(mCloth);
    if (mFixedPointIds.size())
    {
        mCloth->AddFixedVertices(mFixedPointIds);
    }
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

#include "SeObstacle.h"
#include "sim/KinematicBody.h"
tVector CalcNormal(
    const tVector &v0,
    const tVector &v1,
    const tVector &v2)
{

    return ((v1 - v0).cross3(v2 - v1)).normalized();
}
void cLinctexScene::CreateObstacle(const Json::Value &conf)
{
    cSimScene::CreateObstacle(conf);
    std::vector<Int3> se_triangles(0);
    std::vector<Float3> se_positions(0);
    std::vector<Float3> se_normals(0);
    {
        const auto &v_array = mObstacle->GetVertexArray();
        // const auto &e_array = mObstacle->GetEdgeArray();
        const auto &t_array = mObstacle->GetTriangleArray();

        tEigenArr<tVector> v_normal_array(v_array.size(), tVector::Zero());
        std::vector<int> v_normal_array_count(v_array.size(), 0);

        //  init triangle and calculate vertex normal
        for (int i = 0; i < t_array.size(); i++)
        {
            se_triangles.push_back(Int3(
                t_array[i]->mId0,
                t_array[i]->mId1,
                t_array[i]->mId2));
            tVector f_normal = CalcNormal(
                v_array[t_array[i]->mId0]->mPos,
                v_array[t_array[i]->mId1]->mPos,
                v_array[t_array[i]->mId2]->mPos);
            {
                v_normal_array[t_array[i]->mId0] += f_normal;
                v_normal_array[t_array[i]->mId1] += f_normal;
                v_normal_array[t_array[i]->mId2] += f_normal;
            }
            {
                v_normal_array_count[t_array[i]->mId0] += 1;
                v_normal_array_count[t_array[i]->mId1] += 1;
                v_normal_array_count[t_array[i]->mId2] += 1;
            }
        }

        // calculate vertices and normals
        for (int i = 0; i < v_array.size(); i++)
        {
            se_positions.push_back(
                Float3(
                    v_array[i]->mPos[0],
                    v_array[i]->mPos[1],
                    v_array[i]->mPos[2]));
            v_normal_array[i] /= v_normal_array_count[i];
            se_normals.push_back(
                Float3(
                    v_normal_array[i][0],
                    v_normal_array[i][1],
                    v_normal_array[i][2]));
        }
    }
    auto obstacle = SeObstacle::Create(se_triangles,
                                       se_positions,
                                       se_normals);
    // std::cout << "old obstacle offset = " << obstacle->GetSurfaceOffset() << std::endl;
    obstacle->SetSurfaceOffset(0);
    // mSeScene->GetSimulationParameters()->
    // this->mCloth->GetSimulationProperties()->
    // std::cout << "new obstacle offset = " << obstacle->GetSurfaceOffset() << std::endl;
    // exit(0);
    mSeScene->AddObstacle(obstacle);
    std::cout << "[debug] add linctex obstacle succ\n";
}

void cLinctexScene::SetSimProperty(const tPhyPropertyPtr &prop)
{
    mClothProp = prop;
    auto phyProp = mCloth->GetPhysicalProperties();
    phyProp->SetStretchWarp(mClothProp->mStretchWarp);
    phyProp->SetStretchWeft(mClothProp->mStretchWeft);
    phyProp->SetBendingWarp(mClothProp->mBendingWarp);
    phyProp->SetBendingWeft(mClothProp->mBendingWeft);
}
tPhyPropertyPtr cLinctexScene::GetSimProperty() const
{
    return mClothProp;
}

/**
 * \brief           Get the feature vector of this cloth
 * 
 *  Current it's all nodal position of current time
*/
const tVectorXd &cLinctexScene::GetClothFeatureVector() const
{
    return mClothFeature;
}

int cLinctexScene::GetClothFeatureSize() const
{
    return mClothFeature.size();
}

/**
 * \brief           Save current simulation correspondence
*/
void cLinctexScene::DumpSimulationData(
    const tVectorXd &simualtion_result,
    const tVectorXd &simulation_property,
    const tVector &init_rot_qua,
    const tVector &init_translation,
    const std::string &filename)
{
    Json::Value export_json;
    export_json["input"] = cJsonUtil::BuildVectorJson(simualtion_result);
    export_json["output"] = cJsonUtil::BuildVectorJson(simulation_property);
    Json::Value extra_info;
    // std::cout << "feature = " << props->BuildFeatureVector().transpose() << std::endl;
    // std::cout << "trans = \n"
    //           << init_trans << std::endl;

    extra_info["init_rot"] = cJsonUtil::BuildVectorJson(init_rot_qua);
    extra_info["init_pos"] = cJsonUtil::BuildVectorJson(init_translation);
    export_json["extra_info"] = extra_info;
    cJsonUtil::WriteJson(filename, export_json);
    std::cout << "[debug] save data to " << filename << std::endl;
}
/**
 * \brief               Init the feature vector
*/
void cLinctexScene::InitClothFeatureVector()
{
    mClothFeature.noalias() = tVectorXd::Zero(mVertexArray.size() * 3);
    UpdateClothFeatureVector();
}

/**
 * \brief               Calculate the feature vector of the cloth
*/
void cLinctexScene::UpdateClothFeatureVector()
{
    for (int i = 0; i < mVertexArray.size(); i++)
    {
        mClothFeature.segment(3 * i, 3).noalias() = mVertexArray[i]->mPos.segment(0, 3);
    }
}

/**
 * \brief               Update nodal position from a vector
*/
void cLinctexScene::UpdateCurNodalPosition(const tVectorXd &xcur)
{
    cSimScene::UpdateCurNodalPosition(xcur);
    if (mCloth)
    {
        std::vector<Float3> pos(0);
        for (int i = 0; i < mVertexArray.size(); i++)
        {
            pos.push_back(
                Float3(
                    xcur[3 * i + 0],
                    xcur[3 * i + 1],
                    xcur[3 * i + 2]));
        }
        mCloth->SetPositions(pos);
    }
}

/**
 * \brief               Init the geometry and set the init positions
*/
void cLinctexScene::InitGeometry(const Json::Value &conf)
{
    cSimScene::InitGeometry(conf);
    std::string init_geo = cJsonUtil::ParseAsString("cloth_init_nodal_position", conf);
    bool is_illegal = true;
    if (cFileUtil::ExistsFile(init_geo) == true)
    {
        Json::Value value;
        cJsonUtil::LoadJson(init_geo, value);
        tVectorXd vec = cJsonUtil::ReadVectorJson(
            cJsonUtil::ParseAsValue("input", value));
        // std::cout << "vec size = " << vec.size();
        if (vec.size() == GetNumOfFreedom())
        {
            mClothInitPos = vec;
            mXcur = vec;
            UpdateCurNodalPosition(mXcur);
            std::cout << "[debug] set init vec from " << init_geo << std::endl;
        }
        else
        {
            is_illegal = false;
        }
    }
    else
    {
        is_illegal = false;
    }

    // 1. check whehter it's illegal
    if (is_illegal)
    {
        std::cout << "[warn] init geometry file " << init_geo << "is illegal, ignore\n";
    }
}
/**
 * \brief           apply the transform to the cloth
*/
void cLinctexScene::ApplyTransform(const tMatrix &trans)
{
    for (int i = 0; i < mVertexArray.size(); i++)
    {

        tVector cur_pos = tVector::Ones();
        cur_pos.segment(0, 3).noalias() = mXcur.segment(i * 3, 3);
        mXcur.segment(i * 3, 3).noalias() = (trans * cur_pos).segment(0, 3);
    }
    UpdateCurNodalPosition(mXcur);
}
#include "GLFW/glfw3.h"
void cLinctexScene::Key(int key, int scancode, int action, int mods)
{
    cSimScene::Key(key, scancode, action, mods);
    if (key == GLFW_KEY_S && action == GLFW_PRESS)
    {
        cLinctexScene::DumpSimulationData(
            GetClothFeatureVector(),
            mClothProp->BuildFeatureVector(),
            tVector::Zero(),
            tVector::Zero(),
            "tmp.json");
    }
}
#endif