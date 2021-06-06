#include "utils/MathUtil.h"
#include "utils/JsonUtil.h"
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
#include "sim/ClothProperty.h"
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

void cLinctexScene::UpdateSubstep() { SIM_ERROR("should not be called"); }

void cLinctexScene::InitConstraint(const Json::Value &value)
{
    cSimScene::InitConstraint(value);
}

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

        mEnableDumpGeometryInfo =
            cJsonUtil::ParseAsBool("enable_dump_geometry_info", root);
        if (mEnableDumpGeometryInfo)
        {
            mDumpGeometryInfoPath =
                cJsonUtil::ParseAsString("dump_triangle_info_path", root);
        }
    }

    InitGeometry(root);
    InitRaycaster();
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
    std::cout << "[debug] se col thickness = "
              << cloth_sim_prop->GetCollisionThickness() << std::endl;
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
        std::cout << "iter " << mNetworkInfer_CurIter
                  << " cur norm = " << cur_norm << std::endl;
        if (cur_norm > mNetworkInfer_ConvThreshold ||
            mNetworkInfer_CurIter < mNetworkInfer_MinIter)
        {
            convergence = false;
        }
    }
    mPreviosFeature = mClothFeature;
    if (convergence)
    {
        std::cout << "exit, save current result to " << mNetworkInfer_OutputPath
                  << std::endl;
        DumpSimulationData(mClothFeature,
                                          mClothProp->BuildFullFeatureVector(),
                                          // tVector::Zero(),
                                          // tVector::Zero(),
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
    // std::cout << "update " << dt << std::endl;
    // SIM_ERROR("update hasn't been supported");
    // mSeScene->
    UpdatePerturb();
    if (mSeScene->Capture())
    {
        mSeScene->AcquirePositions();
        auto &pos = mCloth->FetchPositions();
        for (int i = 0; i < mVertexArray.size(); i++)
        {
            mVertexArray[i]->mPos.noalias() =
                tVector(pos[i][0], pos[i][1], pos[i][2], 1);
        }
        UpdateClothFeatureVector();

        if (mEnableNetworkInferenceMode == true)
            NetworkInferenceFunction();
    }
    // std::cout << "[debug] cloth feature norm = " <<
    // GetClothFeatureVector().norm() << std::endl; else
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
        // printf("[debug] triangle %d: vertices: %d, %d, %d\n", i, tri->mId0,
        // tri->mId1, tri->mId2);
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

void cLinctexScene::ReadVertexPosFromEngine() {}

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
        mDragPt = mCloth->AddDraggedPoint(
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
        mCloth->Remove(mDragPt);
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

void cLinctexScene::SetSimProperty(const tPhyPropertyPtr &prop)
{
    mClothProp = prop;
    auto phyProp = mCloth->GetPhysicalProperties();
    phyProp->SetStretchWarp(mClothProp->mStretchWarp);
    phyProp->SetStretchWeft(mClothProp->mStretchWeft);
    phyProp->SetBendingWarp(mClothProp->mBendingWarp);
    phyProp->SetBendingWeft(mClothProp->mBendingWeft);
    phyProp->SetBendingBias(mClothProp->mBendingBias);
}
tPhyPropertyPtr cLinctexScene::GetSimProperty() const { return mClothProp; }

/**
 * \brief           Get the feature vector of this cloth
 *
 *  Current it's all nodal position of current time
 */
const tVectorXd &cLinctexScene::GetClothFeatureVector() const
{
    return mClothFeature;
}

int cLinctexScene::GetClothFeatureSize() const { return mClothFeature.size(); }

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
        mClothFeature.segment(3 * i, 3).noalias() =
            mVertexArray[i]->mPos.segment(0, 3);
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
                Float3(xcur[3 * i + 0], xcur[3 * i + 1], xcur[3 * i + 2]));
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
    std::string init_geo =
        cJsonUtil::ParseAsString("cloth_init_nodal_position", conf);
    bool is_illegal = true;
    if (cFileUtil::ExistsFile(init_geo) == true)
    {
        Json::Value value;
        cJsonUtil::LoadJson(init_geo, value);
        tVectorXd vec =
            cJsonUtil::ReadVectorJson(cJsonUtil::ParseAsValue("input", value));
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
        std::cout << "[warn] init geometry file " << init_geo
                  << "is illegal, ignore\n";
    }

    if (mEnableDumpGeometryInfo == true)
    {
        cTriangulator::SaveGeometry(this->mVertexArray, this->mEdgeArray,
                                    this->mTriangleArray,
                                    this->mDumpGeometryInfoPath);
    }
}
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
        DumpSimulationData(GetClothFeatureVector(),
                                          mClothProp->BuildFullFeatureVector(),
                                          // tVector::Zero(),
                                          // tVector::Zero(),
                                          "tmp.json");
    }
}

/**
 * \brief           Calculate the center of mass
 */
tVector cLinctexScene::CalcCOM() const
{
    tVector com = tVector::Zero();
    double total_mass = 0;
    for (auto &x : mVertexArray)
    {
        total_mass += x->mMass;
        com += x->mMass * x->mPos;
        // std::cout << "pos = " << x->mPos.transpose() << std::endl;
    }
    com /= total_mass;
    com[3] = 1;
    return com;
}

void cLinctexScene::End() { this->mSeScene->End(); }
#include <cmath>
#include <math.h>

/**
 * \brief           Apply random gaussian noise, point-wise
 */
void cLinctexScene::ApplyNoise(bool enable_y_random_rotation,
                               double &rotation_angle, bool enable_y_random_pos,
                               const double random_ypos_std)
{
    rotation_angle = 0;
    if (enable_y_random_rotation == true)
    {
        rotation_angle = cMathUtil::RandDouble(0, 2 * M_PI);
        // std::cout << "rotation angle = " << rotation_angle << std::endl;
    }

    tMatrix mat = cMathUtil::EulerAnglesToRotMat(
        tVector(0, rotation_angle, 0, 0), eRotationOrder::XYZ);
    for (int i = 0; i < mVertexArray.size(); i++)
    {
        // 1. apply rotation
        tVector cur_pos = tVector::Ones();
        cur_pos.segment(0, 3).noalias() = mXcur.segment(i * 3, 3);
        mXcur.segment(i * 3, 3).noalias() = (mat * cur_pos).segment(0, 3);

        // 2. apply noise translation
        if (enable_y_random_pos == true)
        {
            mXcur[i * 3 + 1] += cMathUtil::RandDoubleNorm(0, random_ypos_std);
        }

        // if (enable_y_random_pos)
        // {
        // }
        // exit(0);
    }

    // 3. final update
    UpdateCurNodalPosition(mXcur);
}

/**
 * s\brief          Apply fold noise by given an axis and the parabola coef "a"
 */
void cLinctexScene::ApplyFoldNoise(const tVector3d &principle_noise,
                                   const double a)
{
    // tVector3d old_pos = tVector3d(0, 0, 0);
    // tVector3d dir_origin = tVector3d(0, 1, 1);
    // tVector3d dir_end = tVector3d(1, 1, 1);
    // std::cout << "dist = " << cMathUtil::CalcDistanceFromPointToLine(old_pos,
    // dir_origin, dir_end) << std::endl;
    SIM_ASSERT(std::fabs(principle_noise.norm() - 1) < 1e-10);
    tVector3d COM = this->CalcCOM().segment(0, 3);
    // std::cout << "COM = " << COM.transpose() << std::endl;
    // std::cout << "principle noise dir = " << principle_noise.transpose() <<
    // std::endl; int num_of_positive = 0, num_of_negative = 0;
    for (int i = 0; i < mXcur.size() / 3; i++)
    {
        const tVector3d &old_pos = mXcur.segment(i * 3, 3);
        // std::cout << "old pos = " << old_pos.transpose() << std::endl;
        // 1. calculate distance between cur pos and the center line
        double dist = cMathUtil::CalcDistanceFromPointToLine(
            old_pos, COM, COM + principle_noise);
        // std::cout << "dist = " << dist << std::endl;
        // 2. calculate (x, y) pos on projected plane, defined by given
        // principle noise vector 2.1 determine the positive or negative
        tVector3d com_2_pos = old_pos - COM;
        // std::cout << "com 2 pos = " << com_2_pos.transpose() << std::endl;
        tVector3d com_2_pos_parallel_with_principle =
            com_2_pos.dot(principle_noise) * principle_noise;
        // std::cout << "com_2_pos_parallel_with_principle = " <<
        // com_2_pos_parallel_with_principle.transpose() << std::endl;
        tVector3d com_2_pos_proj =
            com_2_pos - com_2_pos_parallel_with_principle;
        // std::cout << "com_2_pos_proj = " << com_2_pos_proj.transpose() <<
        // std::endl;
        int sign =
            (tVector3d(0, 1, 0).cross(com_2_pos_proj)).dot(principle_noise) > 0
                ? 1
                : -1;
        // std::cout << "sign = " << sign << std::endl;

        // 2.2

        double x = sign * dist / (std::sqrt(1 + a * a));
        double y = -std::fabs(a * x);
        tVector3d x_dir = principle_noise.cross(tVector3d(0, 1, 0));
        // std::cout << "x_dir = " << x_dir.transpose() << std::endl;
        // std::cout << "x = " << x << std::endl;
        // std::cout << "y = " << y << std::endl;
        tVector3d new_pos =
            (x * x_dir + y * tVector3d(0, 1, 0) + COM) // projected pos
            + com_2_pos_parallel_with_principle;
        // std::cout << "new_pos = " << new_pos.transpose() << std::endl;
        mXcur.segment(i * 3, 3) = new_pos;
        // exit(0);
    }
    UpdateCurNodalPosition(mXcur);
    // std::cout << num_of_positive << std::endl;
    // std::cout << num_of_negative << std::endl;
}

/**
 * \brief                   Apply multiple folds noise
 * \param num_of_folds      Given number of folds
 */
void cLinctexScene::ApplyMultiFoldsNoise(int num_of_folds, double max_amp)
{
    SIM_ASSERT(num_of_folds >= 2 && num_of_folds <= 10);
    // 1. calculate the fold cycle (theta)
    double theta = 2 * M_PI / num_of_folds;
    // std::cout << "cycle theta = " << theta << std::endl;
    // 2. calculate the fold direction, random the amptitude
    // double st_bias = cMathUtil::RandDoubleNorm(0, theta / 3);
    double st_bias = 0;
    // std::cout << "bias = " << st_bias << std::endl;

    // lying on the XOZ plane
    tEigenArr<tVector2d> fold_directions_array(0);
    std::vector<double> fold_st_angle_array(0);
    std::vector<double> fold_amp_array(0);
    for (int i = 0; i < num_of_folds; i++)
    {

        // 2 * 1 vector
        double angle = theta * i + st_bias;
        angle = cMathUtil::NormalizeAngle(angle);
        // std::cout << "angle " << i << " = " << angle << std::endl;
        // double amp = cMathUtil::RandDouble(0, 0.1); // up to 10 cm amp
        // double amp = 0.1;
        double amp = cMathUtil::RandDouble(0, max_amp);
        // double amp = cMathUtil::RandDoubleNorm(0.1, 0.1);
        tVector fold_dir =
            cMathUtil::AxisAngleToRotmat(tVector(0, 1, 0, 0) * angle) *
            tVector(1, 0, 0, 0);
        // printf("[debug] angle %d = %.3f, dir = ", i, angle);
        fold_st_angle_array.push_back(angle);
        fold_amp_array.push_back(amp);
        // project to XOZ plane
        fold_directions_array.push_back(tVector2d(fold_dir[0], fold_dir[2]));
        // printf("[info] angle %d = %.3f, amp = %.3f\n", i, angle, amp);
        // std::cout << fold_directions_array[fold_directions_array.size() -
        // 1].transpose() << std::endl;
    }

    /*
        3. calculate the noise for each point
            3.1 for each point, calculate the pole coordinate
            3.2 confirm two fold direction
            3.3 calculate the height field for these 2 fold. (clamped cos
       function) 3.4 averaging these 2 values, apply this height
    */
    auto calc_angle_distance = [](double first, double second)
    {
        first = cMathUtil::NormalizeAngle(first);
        second = cMathUtil::NormalizeAngle(second);
        double dist = std::fabs(first - second);
        if (dist > M_PI)
            return 2 * M_PI - dist;
        else
            return dist;
    };

    std::vector<int> times(num_of_folds, 0);
    tVector com = CalcCOM();
    // int idx = 0;
    for (int v_id = 0; v_id < mVertexArray.size(); v_id++)
    {
        auto &v = mVertexArray[v_id];
        // project the nodal vector to XOZ plane, calculate the "angle"
        tVector node_vec = v->mPos - com;
        // double theta = 2 * M_PI / mVertexArray.size() * (idx++);
        // tVector node_vec =
        //     tVector(
        //         std::cos(theta),
        //         0,
        //         std::sin(theta), 0);
        node_vec[1] = 0;
        node_vec.normalize();
        // std::cout << tVector(1, 0, 0, 0).cross3(tVector(0, 0, -1, 0)) <<
        // std::endl; exit(0);
        tVector res = cMathUtil::CalcAxisAngleFromOneVectorToAnother(
            tVector(1, 0, 0, 0), node_vec);
        if (res[1] < 0)
        {
            res = tVector(0, 2 * M_PI + res[1], 0, 0);
        }
        // {
        //     tVector residual = tVector(0, 1, 0, 0) * res.norm() - res;
        //     SIM_ASSERT(residual.norm() < 1e-6);
        //     if (residual.norm() > 1e-6)
        //     {
        //         std::cout << "node_vec = " << node_vec.transpose() <<
        //         std::endl; std::cout << "res = " << res.transpose() <<
        //         std::endl; exit(1);
        //     }
        // }
        double cur_angle = cMathUtil::NormalizeAngle(res[1]);
        // double cur_angle = 3;
        // double cur_angle = 1.24081;
        // std::cout << "for point " << node_vec.transpose() << " its angle = "
        // << cur_angle << std::endl;
        int interval0 = -1, interval1 = -1;
        for (int i = 0; i < num_of_folds; i++)
        {
            // for fold 1
            int fold0_id = i, fold1_id = (i + 1) % num_of_folds;
            double angle0 = fold_st_angle_array[fold0_id],
                   angle1 = fold_st_angle_array[fold1_id];

            // if the value is on the boundary, include them
            if (std::fabs(angle0 - cur_angle) < 1e-6 ||
                std::fabs(angle1 - cur_angle) < 1e-6)
            {
                interval0 = fold0_id;
                interval1 = fold1_id;
                break;
            }
            else
            {
                double interval = calc_angle_distance(angle0, angle1);
                if (calc_angle_distance(angle0, cur_angle) < interval &&
                    calc_angle_distance(angle1, cur_angle) < interval)
                {
                    interval0 = fold0_id;
                    interval1 = fold1_id;
                    break;
                }
            }
        }

        if (interval0 == -1 || interval1 == -1)
        {
            std::cout << "[error] for angle " << cur_angle
                      << " failed to judge the interval. cur interval are:";
            for (auto &x : fold_st_angle_array)
                std::cout << x << " ";
            std::cout << std::endl;
            exit(0);
        }
        else
        {
            times[interval0] += 1;

            double amp0 = fold_amp_array[interval0],
                   amp1 = fold_amp_array[interval1];
            double int_angle0 = fold_st_angle_array[interval0],
                   int_angle1 = fold_st_angle_array[interval1];
            double angle_with0 = calc_angle_distance(cur_angle, int_angle0);
            double angle_with1 = calc_angle_distance(cur_angle, int_angle1);
            double bias = 0;
            if (angle_with0 < theta / 2)
            {
                bias += std::cos(angle_with0 / (theta / 2) * M_PI) * amp0;
            }
            else
            {
                bias += -amp0 * std::pow(angle_with1 / theta, 2);
            }
            if (angle_with1 < theta / 2)
            {
                bias += std::cos(angle_with1 / (theta / 2) * M_PI) * amp1;
            }
            else
            {
                bias += -amp1 * std::pow(angle_with0 / theta, 2);
            }

            // remove stretch
            double raw_length = (v->mPos - com).segment(0, 3).norm();
            double max_length = mClothWidth / 2 * std::sqrt(2);
            bias *= std::pow(raw_length / max_length, 3);
            mXcur[3 * v_id + 1] += bias;
            // tVector3d ref = mXcur.segment(3 * (v_id - 1), 3);
            // tVector3d cur = mXcur.segment(3 * (v_id), 3);

            // mXcur.segment(3 * v_id, 3) = (cur - ref).normalized() *
            // (v->mPos.segment(0, 3) - ref).norm() + ref.segment(0, 3);
            // v->mPos[1] ;
            // printf("[info] angle %.4f is in [%.3f, %.3f]\n", cur_angle,
            // fold_st_angle_array[interval0],
            //        fold_st_angle_array[interval1]);
        }
    }
    UpdateCurNodalPosition(mXcur);
    // for (auto &x : times)
    // {
    //     std::cout << x << std::endl;
    // }
    // exit(0);
}
#endif
