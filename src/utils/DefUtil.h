#include <memory>

// Predefined
#define SIM_DECLARE_PTR(ClassObject)                                           \
    using ClassObject##Ptr = std::shared_ptr<ClassObject>;

#define SIM_DECLARE_CLASS_AND_PTR(ClassObject)                                 \
    class ClassObject;                                                         \
    SIM_DECLARE_PTR(ClassObject);

#define SIM_DECLARE_STRUCT_AND_PTR(ClassObject)                                \
    struct ClassObject;                                                        \
    SIM_DECLARE_PTR(ClassObject);

#define RENDERING_SIZE_PER_VERTICE (8)