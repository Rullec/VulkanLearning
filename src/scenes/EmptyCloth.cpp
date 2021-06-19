#include "scenes/EmptyCloth.h"
#include "geometries/Triangulator.h"
cEmptyCloth::cEmptyCloth(int id_) : cBaseCloth(eClothType::EMPTY_CLOTH, id_) {}

cEmptyCloth::~cEmptyCloth() {}

void cEmptyCloth::UpdatePos(double dt) {}

void cEmptyCloth::LoadGeometry(std::string geo_info_path)
{
    cTriangulator::LoadGeometry(mVertexArray, mEdgeArray, mTriangleArray,
                                geo_info_path);
}