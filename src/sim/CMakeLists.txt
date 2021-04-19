set(
    SIM_SRC
    BaseObject.cpp
    KinematicBody.cpp
    ClothPropertyManager.cpp
    ClothProperty.cpp
)

add_library(sim_lib ${SIM_SRC})