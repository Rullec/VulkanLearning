#pragma once
#include "SimScene.h"
#include <vector>

class cMeshVisScene : public cSimScene
{
public:
    inline static const std::string MESH_DATA_KEY = "mesh_data";

    cMeshVisScene();
    ~cMeshVisScene();
    virtual void Init(const std::string &conf_path) override;
    virtual void Update(double dt) override;
    virtual void Key(int key, int scanecode, int action, int mods) override;

protected:
    int mCurMeshId;
    std::string mMeshDataDir;
    std::vector<std::string> mMeshDataList;
    virtual void UpdateSubstep() override;
    void SetMeshData(int id);
};