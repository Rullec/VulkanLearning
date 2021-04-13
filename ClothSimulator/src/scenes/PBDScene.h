#pragma once
#include "SimScene.h"

/**
 * \brief           using trimesh to modeling the cloth
 * 
 *  1. Positon based method 
 *  2. baraff 98 siggraph method
*/

struct tTriangle;
struct tEdge;
struct tRay;
class cPBDScene : public cSimScene
{
public:
    explicit cPBDScene();
    virtual void Init(const std::string &conf_path) override;
    ~cPBDScene();

protected:
    const int mMaxDrawRayDebug = 100;
    // std::vector<tEdge *> mEdgeArray;
    tVectorXd mVcur; // velocity vector
    virtual void InitGeometry(const Json::Value &conf) override final;
    // void InitGeometryUniformSquare();
    // void InitGeometrySkewTriangle();
    // void InitGeometryRegularTriangle();
    virtual void InitConstraint(const Json::Value &root) override final;
    virtual void UpdateSubstep() override final;
    virtual void CalcTriangleDrawBuffer() override final;
    virtual void CalcEdgesDrawBuffer() override final;

    // PBD methods
    int mItersPBD;
    double mStiffnessPBD;
    bool mEnableParallelPBD;                 // enable parallel pbd
    bool mEnableBendingPBD;                  // enable bending constraint
    double mBendingStiffnessPBD;             // bending constraint stiffness PBD
    tEigenArr<tVector> mBendingMatrixKArray; // the array of bending matrix "K" for inextensible surface
    // std::vector<std::vector<int>> mColorGroupPBD; // divide all constraints into several groups, which there is not shared vertices in a same group

    // void InitColorGroupPBD();
    void UpdateSubstepPBD();
    void UpdateVelAndPosUnconstrained(const tVectorXd &fext);
    virtual void CalcExtForce(tVectorXd &ext_force) const override;
    // void ConstraintSetupPBD();
    void ConstraintProcessPBD();
    void PostProcessPBD();

    void StretchConstraintProcessPBD(double final_k);
    void BendingConstraintProcessPBD(double final_k);
    void InitBendingMatrixPBD();
    // Projective dynamic
    void UpdateSubstepProjDyn();
};