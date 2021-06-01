
# device manager
pybind11_add_module(device_manager ./device_manager_pybind_export.cpp ./DeviceManager.cpp ./AxonManager.cpp KinectManager.cpp)
target_link_libraries(device_manager PRIVATE OpenNI2)
target_link_libraries(device_manager PRIVATE utils_lib)
target_link_libraries(device_manager PRIVATE k4a)

move_pybind11_release_to_src(learning device_manager)