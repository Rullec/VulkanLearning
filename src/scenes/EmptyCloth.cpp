#include "scenes/EmptyCloth.h"
#include "geometries/Triangulator.h"
cEmptyCloth::cEmptyCloth() : cBaseCloth(eClothType::EMPTY_CLOTH) {}

cEmptyCloth::~cEmptyCloth() {}

void cEmptyCloth::UpdatePos(double dt) {}

void cEmptyCloth::LoadGeometry(std::string geo_info_path)
{
    cTriangulator::LoadGeometry(mVertexArray, mEdgeArray, mTriangleArray,
                                geo_info_path);
}