#pragma once
#include "sim/cloth/BaseCloth.h"

class cEmptyCloth : public cBaseCloth
{
public:
    explicit cEmptyCloth(int id_);
    ~cEmptyCloth();
    virtual void UpdatePos(double dt) override final;
    virtual void LoadGeometry(std::string geo_info_path);
};