#ifdef _WIN32
#pragma once
#include "sim/cloth/BaseCloth.h"
#include "utils/DefUtil.h"

namespace StyleEngine
{
class SePiece;
class SeScene;
class SeDraggedPoints;
}; // namespace StyleEngine

SIM_DECLARE_CLASS_AND_PTR(tPhyProperty);
class cLinctexCloth : public cBaseCloth
{
public:
    cLinctexCloth();
    virtual ~cLinctexCloth();
    virtual void Init(const Json::Value &conf);
    virtual void Reset() override final;
    virtual void UpdatePos(double dt) override;
    void SetSimProperty(const tPhyPropertyPtr &prop);
    tPhyPropertyPtr GetSimProperty() const;
    std::shared_ptr<StyleEngine::SePiece> GetPiece() const;
    const tVectorXd &GetClothFeatureVector() const;
    tVector CalcCOM() const;
    int GetClothFeatureSize() const;

    // apply the noise
    void ApplyNoise(bool enable_y_random_rotation, double &rotation_angle,
                    bool enable_y_random_pos, const double random_ypos_std);
    void ApplyFoldNoise(const tVector3d &principle_noise, const double a);
    void ApplyMultiFoldsNoise(int num_of_folds, double max_amp);

protected:
    std::shared_ptr<StyleEngine::SePiece> mSeCloth;
    tPhyPropertyPtr mClothProp;   // cloth property
    bool mEnableDumpGeometryInfo; // if true, we save the geometry information
                                  // after the initialization
    std::string mDumpGeometryInfoPath; // save path for initial geometry
    tVectorXd mClothFeature;

    virtual void InitConstraint(const Json::Value &root) override final;
    virtual void InitGeometry(const Json::Value &conf);
    void AddPiece(); // add the simulation data into the se engine

    void InitClothFeatureVector();
    void UpdateClothFeatureVector();

    virtual void SetPos(const tVectorXd &xcur) override final;
};
#endif