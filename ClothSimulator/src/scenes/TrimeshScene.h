#pragma once
#include "SimScene.h"

/**
 * \brief           using trimesh to modeling the cloth
 * 
 *  1. Positon based method 
 *  2. baraff 98 siggraph method
*/
struct tTriangle
{
    explicit tTriangle();
    explicit tTriangle(int a, int b, int c);
    int mId0, mId1, mId2;
};

struct tEdge
{
    tEdge();
    int mId0, mId1;
    double mRawLength;              // raw length of this edge
    bool mIsBoundary;               // does this edge locate in the boundary?
    int mTriangleId0, mTriangleId1; // The indices of the two triangles to which this side belongs. If this edge is a boundary, the mTriangleId1 is -1
};

class cTrimeshScene : public cSimScene
{
public:
    explicit cTrimeshScene();
    virtual void Init(const std::string &conf_path) override;
    ~cTrimeshScene();

protected:
    tEigenArr<tTriangle *> mTriangleArray;
    tEigenArr<tEdge *> mEdgeArray;
    tVectorXd mVcur; // velocity vector
    tVectorXd mInvMassMatrixDiag;

    virtual void InitGeometry() override final;
    virtual void InitConstraint(const Json::Value &root) override final;
    virtual void UpdateSubstep() override final;
    virtual void CalcTriangleDrawBuffer() override final;
    virtual void CalcEdgesDrawBuffer() override final;

    // PBD methods
    int mItersPBD;
    double mStiffnessPBD;
    bool mEnableParallelPBD;                      // enable parallel pbd
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