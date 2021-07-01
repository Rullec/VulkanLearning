add_subdirectory(./json)

if(UNIX)
    find_package(spdlog REQUIRED)
endif()

find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_DIRS})

add_library(utils_lib FileUtil.cpp
JsonUtil.cpp
LogUtil.cpp
MathUtil.cpp
MathUtil2.cpp
Rand.cpp
TimeUtil.cpp
SysUtil.cpp
ObjUtil.cpp
DepthImageSaver.cpp
MPIUtil.cpp
)
target_link_libraries(utils_lib spdlog JSON ${MPI_CXX_LIBRARIES})

# only for linux
if(UNIX AND (NOT DEFINED APPLE))
    target_link_libraries(utils_lib stdc++fs)
endif()