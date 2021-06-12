
if(USE_OPTIX)
cuda_compile_and_embed(embedded_ptx_code OptixDevicePrograms.cu)
endif()

add_library(
    geo_lib
    ${embedded_ptx_code}
    Primitives.cpp
    Triangulator.cpp
    Raycaster.cpp
    OptixRaycaster.cpp
    CollisionDetecter.cpp
    CollisionInfo.cpp
)

if(USE_OPTIX)
target_link_libraries(geo_lib
    ${optix_LIBRARY}
    ${CUDA_LIBRARIES}
    ${CUDA_CUDA_LIBRARY}
  )
endif()