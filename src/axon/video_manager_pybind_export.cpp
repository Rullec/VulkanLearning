#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "VideoManager.h"

namespace py = pybind11;
PYBIND11_MODULE(video_manager, m)
{
    py::class_<cVideoManager>(m, "video_manager")
    .def(py::init<>())
    .def("GetDepthImage", &cVideoManager::GetDepthImage)
    .def("GetDepthUnit_mm", &cVideoManager::GetDepthUnit_mm)
    .def("GetIrImage", &cVideoManager::GetIrImage)
    ;
}