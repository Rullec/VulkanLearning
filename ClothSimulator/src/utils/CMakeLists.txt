add_subdirectory(./json)
find_package(spdlog REQUIRED)
add_library(utils_lib FileUtil.cpp
JsonUtil.cpp
LogUtil.cpp
MathUtil.cpp
Rand.cpp
TimeUtil.cpp
)
target_link_libraries(utils_lib spdlog JSON)