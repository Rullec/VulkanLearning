#include "scenes/LinctexCloth.h"
#include "utils/JsonUtil.h"
#include "utils/MathUtil.h"
#include <iostream>

/**
 * \brief           Save current simulation correspondence
 */
void DumpSimulationData(const tVectorXd &simualtion_result,
                        const tVectorXd &simulation_property,
                        // const tVector &init_rot_qua,
                        // const tVector &init_translation,
                        const std::string &filename)
{
    Json::Value export_json;
    export_json["input"] = cJsonUtil::BuildVectorJson(simualtion_result);
    export_json["output"] = cJsonUtil::BuildVectorJson(simulation_property);
    // Json::Value extra_info;
    // std::cout << "feature = " << props->BuildFullFeatureVector().transpose()
    // << std::endl; std::cout << "trans = \n"
    //           << init_trans << std::endl;

    // extra_info["init_rot"] = cJsonUtil::BuildVectorJson(init_rot_qua);
    // extra_info["init_pos"] = cJsonUtil::BuildVectorJson(init_translation);
    // export_json["extra_info"] = extra_info;
    cJsonUtil::WriteJson(filename, export_json);
    std::cout << "[debug] save data to " << filename << std::endl;
}

void LoadSimulationData(tVectorXd &simualtion_result,
                        tVectorXd &simulation_property,
                        const std::string &filename)
{
    Json::Value root;
    cJsonUtil::LoadJson(filename, root);
    simualtion_result =
        cJsonUtil::ReadVectorJson(cJsonUtil::ParseAsValue("input", root));
    simulation_property =
        cJsonUtil::ReadVectorJson(cJsonUtil::ParseAsValue("output", root));
}

#ifdef _WIN32
#include "Core/SeLogger.h"
#include "LinctexScene.h"
#include "SePhysicalProperties.h"
#include "SePiece.h"
#include "SeScene.h"
#include "SeSceneOptions.h"
#include "SeSimParameters.h"
#include "SeSimulationProperties.h"
#include "geometries/Primitives.h"
#include "geometries/Triangulator.h"
#include "sim/cloth/ClothProperty.h"
#include "utils/FileUtil.h"
#include "utils/LogUtil.h"
#include <chrono> // std::chrono::seconds
#include <thread> // std::this_thread::sleep_for
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

cLinctexScene::~cLinctexScene() {}

#include "utils/JsonUtil.h"
#include "utils/MathUtil.h"
extern const tVector gGravity;

void logging(const char *a, const char *b, int c, SeLogger::Level d,
             const char *e)
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
#include "sim/cloth/BaseCloth.h"
void cLinctexScene::Reset()
{
    mSeScene->End();
    mEngineStart = false;
    mCloth->Reset();
}
void cLinctexScene::Init(const std::string &path)
{
    SeLogger::GetInstance()->RegisterCallback(logging);
    cSimScene::Init(path);
    Json::Value root;
    cJsonUtil::LoadJson(path, root);
    // {
    //     mClothProp = std::make_shared<tPhyProperty>();
    //     mClothProp->Init(root);
    // }
    {
        mEnableNetworkInferenceMode =
            cJsonUtil::ParseAsBool("enable_network_inference_mode", root);
        if (mEnableNetworkInferenceMode == true)
        {
            mNetworkInfer_ConvThreshold = 0;
            mNetworkInfer_OutputPath = "";
            mNetworkInfer_ConvThreshold = cJsonUtil::ParseAsDouble(
                "network_inference_convergence_threshold", root);
            mNetworkInfer_OutputPath =
                cJsonUtil::ParseAsString("network_inference_output_path", root);
            mNetworkInfer_MinIter =
                cJsonUtil::ParseAsInt("network_inference_min_iter", root);
            mPreviosFeature.resize(0);
            mNetworkInfer_CurIter = 0;
            std::cout << "[NN] output path = " << mNetworkInfer_OutputPath
                      << std::endl;
            std::cout << "[NN] conv threshold = " << mNetworkInfer_ConvThreshold
                      << std::endl;
            // exit(0);
        }
    }

    // now the cloth should have been added & inited well
    SIM_ASSERT(mCloth != nullptr);

    // std::cout << "begin to add piece to se scene\n";
    auto new_cloth = GetLinctexCloth();

    mSeScene->AddPiece(new_cloth->GetPiece());

    // init other scene
    auto ptr = mSeScene->GetSimulationParameters();
    ptr->SetGravity(gGravity[1]);
    mEngineStart = false;
}

void cLinctexScene::CreateCloth(const Json::Value &conf)
{
    mCloth = std::make_shared<cLinctexCloth>();
    mCloth->Init(conf);
}

void cLinctexScene::NetworkInferenceFunction()
{
    auto se_cloth = GetLinctexCloth();
    bool convergence = true;

    if (mPreviosFeature.size() != GetNumOfFreedom())
    {
        // the first time
        convergence = false;
    }
    else
    {
        double cur_norm =
            (mPreviosFeature - se_cloth->GetClothFeatureVector()).norm();
        std::cout << "iter " << mNetworkInfer_CurIter
                  << " cur norm = " << cur_norm << std::endl;
        if (cur_norm > mNetworkInfer_ConvThreshold ||
            mNetworkInfer_CurIter < mNetworkInfer_MinIter)
        {
            convergence = false;
        }
    }
    mPreviosFeature = se_cloth->GetClothFeatureVector();
    if (convergence)
    {
        auto prop = se_cloth->GetSimProperty();
        std::cout << "exit, save current result to " << mNetworkInfer_OutputPath
                  << std::endl;
        DumpSimulationData(se_cloth->GetClothFeatureVector(),
                           prop->BuildFullFeatureVector(),
                           mNetworkInfer_OutputPath);
        exit(0);
    }
    mNetworkInfer_CurIter++;
}

void cLinctexScene::PauseSim()
{
    if (mPauseSim == true)
    {
        mSeScene->End();
    }
    else
    {
        mSeScene->Start();
    }
    mPauseSim = !mPauseSim;
}
void cLinctexScene::Update(double dt)
{
    // std::cout << "linctex update\n";
    if (mPauseSim == true)
        return;
    if (mEngineStart == false)
    {
        mEngineStart = true;
        mSeScene->Start();
    }
    UpdatePerturb();
    if (mSeScene->Capture())
    {
        mSeScene->AcquirePositions();
        mCloth->UpdatePos(dt);
        if (mEnableNetworkInferenceMode == true)
            NetworkInferenceFunction();
    }
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
        tVector tar_pos = mPerturb->GetGoalPos();
        auto piece = GetLinctexCloth()->GetPiece();
        mDragPt = piece->AddDraggedPoint(
            bary, Float3(tar_pos[0], tar_pos[1], tar_pos[2]));
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
        GetLinctexCloth()->GetPiece()->Remove(mDragPt);
        mDragPt = nullptr;
    }
}
#include <SeFeatureVertices.h>
void cLinctexScene::UpdatePerturb()
{
    if (this->mDragPt != nullptr)
    {
        // tVector target_pos = mPerturb->CalcPerturbPos() +
        // mPerturb->GetPerturbForce() / 10;
        tVector target_pos = mPerturb->GetGoalPos();

        mDragPt->SetPositions(
            {Float3(target_pos[0], target_pos[1], target_pos[2])});
    }
}

#include "SeObstacle.h"
#include "sim/KinematicBody.h"
tVector CalcNormal(const tVector &v0, const tVector &v1, const tVector &v2)
{

    return ((v1 - v0).cross3(v2 - v1)).normalized();
}
void cLinctexScene::CreateObstacle(const Json::Value &conf)
{
    cSimScene::CreateObstacle(conf);

    for (int i = 0; i < mObstacleList.size(); i++)
    {
        auto &x = mObstacleList[i];
        std::vector<Int3> se_triangles(0);
        std::vector<Float3> se_positions(0);
        std::vector<Float3> se_normals(0);
        {
            const auto &v_array = x->GetVertexArray();
            const auto &t_array = x->GetTriangleArray();

            tEigenArr<tVector> v_normal_array(v_array.size(), tVector::Zero());
            std::vector<int> v_normal_array_count(v_array.size(), 0);

            //  init triangle and calculate vertex normal
            for (int i = 0; i < t_array.size(); i++)
            {
                se_triangles.push_back(
                    Int3(t_array[i]->mId0, t_array[i]->mId1, t_array[i]->mId2));
                tVector f_normal = CalcNormal(v_array[t_array[i]->mId0]->mPos,
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

                se_positions.push_back(Float3(v_array[i]->mPos[0],
                                              v_array[i]->mPos[1],
                                              v_array[i]->mPos[2]));
                v_normal_array[i] /= v_normal_array_count[i];
                se_normals.push_back(Float3(v_normal_array[i][0],
                                            v_normal_array[i][1],
                                            v_normal_array[i][2]));
                // std::cout << "v = " << v_array[i]->mPos.transpose() << " n =
                // " << v_normal_array[i].transpose() << std::endl;
            }
        }
        // printf("[debug] obstacle %d, num of triangles %d\n", i,
        // se_triangles.size());
        auto obstacle =
            SeObstacle::Create(se_triangles, se_positions, se_normals);
        // std::cout << "old obstacle offset = " << obstacle->GetSurfaceOffset()
        // << std::endl;
        obstacle->SetSurfaceOffset(0);
        // mSeScene->GetSimulationParameters()->
        // this->mCloth->GetSimulationProperties()->
        // std::cout << "new obstacle offset = " << obstacle->GetSurfaceOffset()
        // << std::endl; exit(0);
        mSeScene->AddObstacle(obstacle);
    }
    std::cout << "[debug] add linctex obstacles succ, num = "
              << mObstacleList.size() << std::endl;
}

// void cLinctexScene::InitGeometry(const Json::Value &conf)
// {

// }
/**
 * \brief           apply the transform to the cloth
 */
// void cLinctexScene::ApplyTransform(const tMatrix &trans)
// {
//     for (int i = 0; i < mVertexArray.size(); i++)
//     {

//         tVector cur_pos = tVector::Ones();
//         cur_pos.segment(0, 3).noalias() = mXcur.segment(i * 3, 3);
//         mXcur.segment(i * 3, 3).noalias() = (trans * cur_pos).segment(0, 3);
//     }
//     UpdateCurNodalPosition(mXcur);
// }
#include "GLFW/glfw3.h"
void cLinctexScene::Key(int key, int scancode, int action, int mods)
{
    cSimScene::Key(key, scancode, action, mods);
    if (key == GLFW_KEY_S && action == GLFW_PRESS)
    {
        DumpSimulationData(
            GetLinctexCloth()->GetClothFeatureVector(),
            GetLinctexCloth()->GetSimProperty()->BuildFullFeatureVector(),
            // tVector::Zero(),
            // tVector::Zero(),
            "tmp.json");
    }
}

void cLinctexScene::End() { this->mSeScene->End(); }

cLinctexClothPtr cLinctexScene::GetLinctexCloth() const
{
    auto new_ptr = std::dynamic_pointer_cast<cLinctexCloth>(mCloth);
    SIM_ASSERT(new_ptr != nullptr);
    return new_ptr;
}
#endif
