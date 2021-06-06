set(
    SIM_SRC
    BaseObject.cpp
    KinematicBody.cpp
    ClothPropertyManager.cpp
    ClothProperty.cpp
    CollisionDetecter.cpp
)

add_library(sim_lib ${SIM_SRC})
target_link_libraries(sim_lib utils_lib)