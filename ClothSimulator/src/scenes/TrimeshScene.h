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
class cTrimeshScene : public cSimScene
{
public:
    explicit cTrimeshScene();
    virtual void Init(const std::string &conf_path) override;
    ~cTrimeshScene();

protected:
    std::vector<tTriangle *> mTriangleArray;
    // std::vector<tEdge *> mEdgeArray;
    tVectorXd mVcur; // velocity vector
    tVectorXd mInvMassMatrixDiag;
    std::string mGeometryType;
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
    bool mEnableParallelPBD; // enable parallel pbd
    // std::vector<std::vector<int>> mColorGroupPBD; // divide all constraints into several groups, which there is not shared vertices in a same group

    // void InitColorGroupPBD();
    void UpdateSubstepPBD();
    void UpdateVelAndPosUnconstrained(const tVectorXd &fext);
    virtual void CalcExtForce(tVectorXd &ext_force) const override;
    // void ConstraintSetupPBD();
    void ConstraintProcessPBD();
    void PostProcessPBD();

    // Projective dynamic
    void UpdateSubstepProjDyn();
};