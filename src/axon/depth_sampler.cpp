#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "DepthSampler.h"

namespace py = pybind11;
PYBIND11_MODULE(depth_sampler, m)
{
    py::class_<cDepthSampler>(m, "depth_camera")
    .def(py::init<>())
    .def("GetDepthImage", &cDepthSampler::GetDepthImage)
    .def("GetDepthUnit_mm", &cDepthSampler::GetDepthUnit_mm)
    ;
}