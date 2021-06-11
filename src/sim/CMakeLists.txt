set(
    SIM_SRC
    BaseObject.cpp
    KinematicBody.cpp
    # CollisionDetecter.cpp
    
    cloth/ClothPropertyManager.cpp
    cloth/ClothProperty.cpp
    cloth/BaseCloth.cpp
    cloth/SemiCloth.cpp
    cloth/PBDCloth.cpp
    cloth/PDCloth.cpp
    cloth/ClothBuilder.cpp
    cloth/ImplicitCloth.cpp
)

add_library(sim_lib ${SIM_SRC})
target_link_libraries(sim_lib utils_lib)