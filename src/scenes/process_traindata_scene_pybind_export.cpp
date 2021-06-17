#include "ProcessTrainDataScene.h"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
PYBIND11_MODULE(process_data_scene, m)
{
    // py::class_<cProcessTrainDataScene>(m, "process_data_scene")
    //     .def(py::init<>())
    //     .def("InitExport", &cProcessTrainDataScene::InitExport);
    py::class_<cProcessTrainDataScene, std::shared_ptr<cProcessTrainDataScene>>(
        m, "process_data_scene")
        .def(py::init<>())
        .def("Init", &cProcessTrainDataScene::Init)
        .def("CalcEmptyDepthImage",
             &cProcessTrainDataScene::CalcEmptyDepthImage)
        .def("GetDepthImageShape", &cProcessTrainDataScene::GetDepthImageShape)
        .def("GetCuttedWindow", &cProcessTrainDataScene::GetCuttedWindow)
        .def("GetEnableOnlyExportingCuttedWindow",
             &cProcessTrainDataScene::GetEnableOnlyExportingCuttedWindow)
        .def("GetResolution", &cProcessTrainDataScene::GetResolution);
}