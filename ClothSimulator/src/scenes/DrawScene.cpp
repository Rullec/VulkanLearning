#include "DrawScene.h"
#include <iostream>
#include "vulkan/vulkan.h"
#include <utils/MathUtil.h>
#include "utils/LogUtil.h"
#include <optional>
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
    vkDestroyDevice(mDevice, nullptr);
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
    SetupDebugMessenger();
    PickPhysicalDevice();
    CreateLogicalDevice();
}

// which queues do we want to support?
struct QueueFamilyIndices
{
    std::optional<uint32_t> graphicsFamily; // here we use optional, because any unit value would be valid and we need to distinguish from non-value case

    /**
     * \brief           Judge: can we use this queue family?
    */
    bool IsComplete() const
    {
        return graphicsFamily.has_value();
    }
};

/**
 * \brief       Given an physical device, find whehter all queue families we want is in it
 *      If an queue family is supported, it has an value
 *      otherwise, its value is <optional>::novalue
*/
QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device)
{
    QueueFamilyIndices indices;
    uint32_t num_of_families = 0;

    // 1. get all queue faimilies on this physical device
    vkGetPhysicalDeviceQueueFamilyProperties(device, &num_of_families, nullptr);
    std::vector<VkQueueFamilyProperties> families(num_of_families); // storage the properties of each queue on this physical device. such as maximium number of queues, types of queues...

    vkGetPhysicalDeviceQueueFamilyProperties(device, &num_of_families, families.data());

    // 2. find a queue to support graphics
    for (int i = 0; i < families.size(); i++)
    {
        const auto &x = families[i];
        //std::cout << i << " queue flags = " << x.queueFlags << std::endl;
        if (x.queueFlags && VK_QUEUE_GRAPHICS_BIT)
        {
            indices.graphicsFamily = i;
        }

        // stop when we get all we want
        if (indices.IsComplete() == true)
            break;
    }

    return indices;
}

/**
 * \brief       Check whether an device is suitable for us
 *      Devices are not equal
*/
bool IsDeviceSuitable(VkPhysicalDevice &dev)
{
    // VkPhysicalDeviceProperties props;
    // VkPhysicalDeviceFeatures feas;
    // vkGetPhysicalDeviceProperties(dev, &props);

    // vkGetPhysicalDeviceFeatures(dev, &feas);

    // // discrete GPU and geo shader support?
    // return props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && feas.geometryShader;
    // return true;
    // is graphics queue (or any queue we want) families supported on this dev?
    QueueFamilyIndices indices = findQueueFamilies(dev);
    return indices.IsComplete();
}

/**
 * \brief       rating a physical device
*/
int RateDeviceSutability(VkPhysicalDevice &dev)
{
    int score = 0;
    VkPhysicalDeviceProperties props;
    VkPhysicalDeviceFeatures feas;
    vkGetPhysicalDeviceProperties(dev, &props);

    vkGetPhysicalDeviceFeatures(dev, &feas);

    if (IsDeviceSuitable(dev) == false)
    {
        std::cout << "device is not suitable\n";
        return 0;
    }
    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
        score += 1e3;
    if (feas.geometryShader == false)
        return 0;
    return score;
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
        for (int i = 0; i < glfwExtensionCount; i++) {
            std::cout << glfwExtensions[i] << std::endl;
        }
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

/**
 * \brief           set the messege callback for validation layer (not impl yet)
*/
void cDrawScene::SetupDebugMessenger()
{
}

/**
 * \brief           After the initializetion of device, create and select a physical device
 * 
 *      the physical device will be stored in the `VkPhysicalDevice`. they will be released along with the `VkDevice`, do not need to release manually
*/
#include <map>
void cDrawScene::PickPhysicalDevice()
{
    mPhysicalDevice = VK_NULL_HANDLE;

    // get the number of physical devices
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(mInstance, &deviceCount, nullptr);
    SIM_ASSERT(deviceCount != 0);

    // get all physical devices
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(mInstance, &deviceCount, devices.data());

    // rate all phy devices
    std::multimap<int, VkPhysicalDevice> candidates;
    for (int i = 0; i < devices.size(); i++)
    {
        auto x = devices[i];
        int score = RateDeviceSutability(x);
        //std::cout << "device " << i << " score " << score << std::endl;
        candidates.insert(std::make_pair(RateDeviceSutability(x), x));
    }
    SIM_ASSERT(candidates.size() > 0);
    // fetch the highest score physical device
    SIM_ASSERT(candidates.begin()->first > 0);
    //std::cout << candidates.begin()->first << std::endl;
    mPhysicalDevice = candidates.begin()->second;
    //std::cout << "[debug] pick up physical device done, device =  " << mPhysicalDevice << " score = " << candidates.begin()->first << std::endl;
    SIM_ASSERT(mPhysicalDevice != VK_NULL_HANDLE);
}

/**
 * \brief           Create the logical device from the given physical device
 * 
 *      write the member "VkDevice mDevice"
*/
void cDrawScene::CreateLogicalDevice()
{
    // 1. fetch all supported queue families from the physical device
    QueueFamilyIndices indices = findQueueFamilies(mPhysicalDevice);
    //std::cout << "physical score = " << RateDeviceSutability(mPhysicalDevice) << std::endl;
    //std::cout << "has value = " << indices.graphicsFamily.has_value() << std::endl;
    //exit(0);
    // 2. set the queue family info into the physical device
    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = indices.graphicsFamily.value();
    queueCreateInfo.queueCount = 1;
    float queuePriority = 1.0f;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    // 3.
    VkPhysicalDeviceFeatures deviceFeatures{};

    // 4. create logical device
    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pQueueCreateInfos = &queueCreateInfo;
    createInfo.queueCreateInfoCount = 1;
    createInfo.pEnabledFeatures = &deviceFeatures;

    {
        // we don't need any more extensions here nowI
        {
            // set extensions count in the DeviceCreateInfo
            createInfo.enabledExtensionCount = 0;
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
    }

    if (vkCreateDevice(mPhysicalDevice, &createInfo, nullptr, &mDevice) != VK_SUCCESS)
    {
        SIM_ERROR("create logic device failed");
    }

    // get the queue handles
    vkGetDeviceQueue(mDevice, indices.graphicsFamily.value(), 0, &mGraphicsQueue);

}