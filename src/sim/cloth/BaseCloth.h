#pragma once
#include "geometries/Primitives.h"
#include "sim/BaseObject.h"
#include "utils/DefUtil.h"

enum eClothType
{
    SEMI_IMPLICIT_CLOTH = 0,
    IMPLICIT_CLOTH,
    PBD_CLOTH,
    PD_CLOTH,
    LINCTEX_CLOTH,
    EMPTY_CLOTH, // cannot be simulated
    FEM_CLOTH,
    NUM_OF_CLOTH_TYPE
};

struct tPerturb;
SIM_DECLARE_CLASS_AND_PTR(cCollisionDetecter);
class cBaseCloth : public cBaseObject
{
public:
    inline static const std::string DAMPING_KEY = "damping",
                                    DEFAULT_TIMESTEP_KEY = "default_timestep";
    explicit cBaseCloth(eClothType cloth_type);
    virtual ~cBaseCloth();
    virtual void Init(const Json::Value &conf);
    virtual void Reset();
    virtual void SetCollisionDetecter(cCollisionDetecterPtr);
    virtual void CalcTriangleDrawBuffer(Eigen::Map<tVectorXf> &res,
                                        int &st) const;
    virtual void CalcEdgeDrawBuffer(Eigen::Map<tVectorXf> &res, int &st) const;
    static eClothType BuildClothType(std::string str);
    // virtual void Update(double dt);
    virtual void ClearForce();
    virtual void ApplyPerturb(tPerturb *pert);
    virtual void UpdatePos(double dt) = 0;

    virtual void SetPos(const tVectorXd &newpos);
    virtual const tVectorXd &GetPos() const;
    virtual double GetDefaultTimestep() { return mIdealDefaultTimestep; }

protected:
    eClothType mClothType;
    cCollisionDetecterPtr mColDetecter;
    double mIdealDefaultTimestep; // default substep dt
    double mClothWidth;
    double mClothMass;
    tVectorXd mInvMassMatrixDiag; // diag inv mass matrix
    std::string mGeometryType;
    double mDamping;         // damping coeff
    tVectorXd mIntForce;     // internal force
    tVectorXd mExtForce;     // external force
    tVectorXd mDampingForce; // external force

    tVectorXd mXpre, mXcur; // previous node position & current node position
    std::vector<int> mFixedPointIds; // fixed constraint point
    tVectorXd mClothInitPos;         // init position of the cloth
    virtual void InitGeometry(
        const Json::Value &conf); // discretazation from square cloth to
    virtual void InitConstraint(const Json::Value &root);
    virtual void CalcIntForce(const tVectorXd &xcur,
                              tVectorXd &int_force) const;
    virtual void CalcExtForce(tVectorXd &ext_force) const;
    virtual void CalcDampingForce(const tVectorXd &vel,
                                  tVectorXd &damping) const;
    int GetNumOfFreedom() const;
    void CalcNodePositionVector(tVectorXd &pos) const;
};