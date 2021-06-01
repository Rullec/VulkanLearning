#include "KinectManager.h"
#include "AxonManager.h"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(device_manager, m)
{
    py::class_<cAxonManager>(m, "axon_manager")
        .def(py::init<>())
        .def("GetDepthImage", &cAxonManager::GetDepthImage)
        .def("GetDepthUnit_mm", &cAxonManager::GetDepthUnit_mm)
        .def("GetIrImage", &cAxonManager::GetIrImage)
        .def("GetDepthIntrinsicDistCoef",
             &cAxonManager::GetDepthIntrinsicDistCoef)
        .def("GetDepthIntrinsicMtx", &cAxonManager::GetDepthIntrinsicMtx);
    py::class_<cKinectManager>(m, "kinect_manager")
        .def(py::init<>())
        .def("GetDepthImage", &cKinectManager::GetDepthImage)
        .def("GetDepthUnit_mm", &cKinectManager::GetDepthUnit_mm)
        .def("GetIrImage", &cKinectManager::GetIrImage)
        .def("GetDepthIntrinsicDistCoef",
             &cKinectManager::GetDepthIntrinsicDistCoef)
        .def("GetDepthIntrinsicMtx", &cKinectManager::GetDepthIntrinsicMtx);
}