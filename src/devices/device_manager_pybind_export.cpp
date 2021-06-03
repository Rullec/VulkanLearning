#include "AxonManager.h"
#include "KinectManager.h"
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
        .def("GetDepthIntrinsicDistCoef_sdk",
             &cAxonManager::GetDepthIntrinsicDistCoef_sdk)
        .def("GetDepthIntrinsicDistCoef_self",
             &cAxonManager::GetDepthIntrinsicDistCoef_self)
        .def("GetDepthIntrinsicMtx_sdk",
             &cAxonManager::GetDepthIntrinsicMtx_sdk)
        .def("GetDepthIntrinsicMtx_self",
             &cAxonManager::GetDepthIntrinsicMtx_self);
             
    py::class_<cKinectManager>(m, "kinect_manager")
        .def(py::init<>())
        .def("GetDepthImage", &cKinectManager::GetDepthImage)
        .def("GetDepthUnit_mm", &cKinectManager::GetDepthUnit_mm)
        .def("GetIrImage", &cKinectManager::GetIrImage)
        .def("GetDepthIntrinsicDistCoef_sdk",
             &cKinectManager::GetDepthIntrinsicDistCoef_sdk)
        .def("GetDepthIntrinsicMtx_sdk",
             &cKinectManager::GetDepthIntrinsicMtx_sdk)
        .def("GetDepthIntrinsicDistCoef_self",
             &cKinectManager::GetDepthIntrinsicDistCoef_self)
        .def("GetDepthIntrinsicMtx_self",
             &cKinectManager::GetDepthIntrinsicMtx_self);
}