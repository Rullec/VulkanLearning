#include "DrawScene.h"
#include <iostream>
#include "vulkan/vulkan.h"
#include <utils/MathUtil.h>
#include "utils/LogUtil.h"
#include "GLFW/glfw3.h"

const std::vector<const char *> validationLayers = {
    "VK_LAYER_KHRONOS_validation"};
#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

cDrawScene::cDrawScene()
{
    mInstance = nullptr;
}

cDrawScene::~cDrawScene()
{
    vkDestroyInstance(mInstance, nullptr);
}

/**
 * \brief       Init vulkan and other stuff
*/
void cDrawScene::Init()
{
    InitVulkan();
    // uint32_t extensionCount = 0;
    // vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

    // std::cout << extensionCount << " extensions supported" << std::endl;
}

/**
 * \brief           Update the render
*/
void cDrawScene::Update(double dt)
{
}

/**
 * \brief           Reset the whole scene
*/
void cDrawScene::Reset()
{
}

/**
 * \brief           Do initialization for vulkan
*/
void cDrawScene::InitVulkan()
{
    CreateInstance();
}

/**
 * \brief           Create Vulkan Instance
 *      create the application appInfo struct and write down some value
*/
void cDrawScene::CreateInstance()
{
    // 1. check validation layer enable & supported?
    if (enableValidationLayers == true && CheckValidationLayerSupport() == false)
    {
        SIM_ERROR("Validation Layers is not supported");
    }
    // 2. application info
    VkApplicationInfo appInfo{};                        // it contains a pNext member for further extension
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO; // type should be set manually
    appInfo.pApplicationName = "ClothSim";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    // Lots' of information are passed by structure instead of function parameters
    // tell the Vulkan driver whcih global extensions and vlidation layers we want to use
    // it effects the entire program.
    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO; // create appInfo
    createInfo.pApplicationInfo = &appInfo;
    // createInfo.pApplicationInfo = nullptr;

    {
        // vulkan is platform agnostic, we need to know what extension glfw init, in order to pass it to the vulkan
        uint32_t glfwExtensionCount = 0;
        const char **glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        createInfo.enabledExtensionCount = glfwExtensionCount;
        createInfo.ppEnabledExtensionNames = glfwExtensions;
    }

    if (enableValidationLayers == true)
    {
        // how many layers do we want to enable?
        createInfo.enabledLayerCount = validationLayers.size();
        // the layer name
        createInfo.ppEnabledLayerNames = validationLayers.data();
    }
    else
    {
        createInfo.enabledLayerCount = 0;
    }

    // given the instance info, create the instance
    SIM_ASSERT(vkCreateInstance(&createInfo, nullptr, &mInstance) == VK_SUCCESS);
    SIM_INFO("CreateInstance succ");
    //CheckAvaliableExtensions();
    // exit(0);
}

/**
 * \brief           Check avaiable extensions
*/
void cDrawScene::CheckAvaliableExtensions() const
{
    uint32_t size;
    vkEnumerateInstanceExtensionProperties(nullptr, &size, nullptr);

    std::vector<VkExtensionProperties> extension(size);
    vkEnumerateInstanceExtensionProperties(nullptr, &size, extension.data());
    for (auto &x : extension)
    {
        std::cout << x.extensionName << " supported\n";
    }
    // exit(0);
}

/**
 * \brief           Check whether validation layer is supported
*/
bool cDrawScene::CheckValidationLayerSupport() const
{
    // 1. get all supported layers
    uint32_t num_of_layer;
    vkEnumerateInstanceLayerProperties(&num_of_layer, nullptr);

    std::vector<VkLayerProperties> props(num_of_layer);
    vkEnumerateInstanceLayerProperties(&num_of_layer, props.data());

    // 2. check whether the validation layer is in it
    bool supported = false;
    for (auto &support_name : props)
    {
        //std::cout << "supported = " << support_name.layerName << std::endl;
        for (auto &requrested : validationLayers)
        {

            if (strcmp(support_name.layerName, requrested) == 0)
            {
                supported = true;
                //std::cout << "validation supported";
                break;
            }
        }
    }

    // 3. return the result
    return supported;
    //return false;
}