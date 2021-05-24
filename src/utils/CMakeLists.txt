add_subdirectory(./json)

if(UNIX)
    find_package(spdlog REQUIRED)
endif()

add_library(utils_lib FileUtil.cpp
JsonUtil.cpp
LogUtil.cpp
MathUtil.cpp
MathUtil2.cpp
Rand.cpp
TimeUtil.cpp
ObjUtil.cpp
DepthEXRSaver.cpp
)
target_link_libraries(utils_lib spdlog JSON)

# only for linux
if(UNIX AND (NOT DEFINED APPLE))
    target_link_libraries(utils_lib stdc++fs)
endif()