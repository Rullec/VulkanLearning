#include "DrawScene.h"
#include <iostream>
#include <set>
#include "vulkan/vulkan.h"
#include <utils/MathUtil.h>
#include "utils/LogUtil.h"
#include <optional>
#define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_INCLUDE_VULKAN
#include "GLFW/glfw3.h"
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

const std::vector<const char *> validationLayers = {
    "VK_LAYER_KHRONOS_validation"};
const std::vector<const char *> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME};
extern GLFWwindow* window;

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
    vkDestroySwapchainKHR(mDevice, mSwapChain, nullptr);
    vkDestroySurfaceKHR(mInstance, mSurface, nullptr);
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
    CreateSurface();
    PickPhysicalDevice();
    CreateLogicalDevice();
    CreateSwapChain();
}

// which queues do we want to support?
struct QueueFamilyIndices
{
    std::optional<uint32_t> graphicsFamily; // here we use optional, because any unit value would be valid and we need to distinguish from non-value case
    std::optional<uint32_t> presentFamily;  // the ability to show the image to the screen
    /**
     * \brief           Judge: can we use this queue family?
    */
    bool IsComplete() const
    {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

/**
 * \brief       Given an physical device, find whehter all queue families we want is in it
 *      If an queue family is supported, it has an value
 *      otherwise, its value is <optional>::novalue
*/
QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device, VkSurfaceKHR surface)
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

        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
        //vkGetPhysicalDeviceSurfaceSupportKHR(device, i, mSurface, &presentSupport);
        if (presentSupport == true)
        {
            indices.presentFamily = i;
        }

        // stop when we get all we want
        if (indices.IsComplete() == true)
            break;
    }

    return indices;
}

bool checkDeviceExtensionSupport(VkPhysicalDevice device)
{
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

    std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());
    for (const auto &x : availableExtensions)
    {
        requiredExtensions.erase(x.extensionName);
    }

    // if required extensions are empty, means that all requred extensions are supported, return true;
    return requiredExtensions.empty();
}

/**
 * \brief           the details about the support for swapchain, on current surface and physical device
*/
struct SwapChainSupportDetails
{
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

/**
 * \brief           check whehter swap chain is supported and how it's supported
*/
SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device, VkSurfaceKHR surface)
{
    SwapChainSupportDetails details;

    // 1. capabilities
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

    // 2. formats
    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
    details.formats.resize(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());

    // 3. presentModes
    uint32_t modesCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &modesCount, nullptr);

    details.presentModes.resize(modesCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &modesCount, details.presentModes.data());

    return details;
}

/**
 * \brief       select the best surface format(color channels and types)
*/
VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR> availableFormats)
{
    for (auto &x : availableFormats)
    {
        if (x.format == VK_FORMAT_B8G8R8A8_SRGB && x.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            return x;
    }
    return availableFormats[0];
}

/**
 * \brief       choose the swapchain present mode
 * 
 *          the author thought the "mailbox" which can support the triple buffering is the best option
*/
VkPresentModeKHR chooseSwapPresentKHR(const std::vector<VkPresentModeKHR> &modes)
{
    for (const auto &x : modes)
    {
        if (x == VK_PRESENT_MODE_MAILBOX_KHR)
        {
            return x;
        }
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}

/**
 * \brief       choose the swap extent (framebuffer size)
 * 
*/
VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities)
{
    if (capabilities.currentExtent.width != UINT32_MAX)
    {
        return capabilities.currentExtent;
    }
    else
    {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);

        VkExtent2D actualExtent = {
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height)};

        actualExtent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));
        actualExtent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actualExtent.height));

        return actualExtent;
    }
}

/**
 * \brief       Check whether an device is suitable for us
 *      Devices are not equal
*/
bool IsDeviceSuitable(VkPhysicalDevice &dev, VkSurfaceKHR surface)
{
    // VkPhysicalDeviceProperties props;
    // VkPhysicalDeviceFeatures feas;
    // vkGetPhysicalDeviceProperties(dev, &props);

    // vkGetPhysicalDeviceFeatures(dev, &feas);

    // // discrete GPU and geo shader support?
    // return props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && feas.geometryShader;
    // return true;
    // is graphics queue (or any queue we want) families supported on this dev?
    QueueFamilyIndices indices = findQueueFamilies(dev, surface);

    // if some extensions are not supported, it cannot be a suitable device
    bool extensionsSupported = checkDeviceExtensionSupport(dev);

    bool swapChainAdequate = false;
    if (extensionsSupported)
    {
        auto details = querySwapChainSupport(dev, surface);
        swapChainAdequate = details.formats.empty() == false && (details.presentModes.empty()) == false;
    }
    return indices.IsComplete() && extensionsSupported && swapChainAdequate;
}

/**
 * \brief       rating a physical device
*/
int RateDeviceSutability(VkPhysicalDevice &dev, VkSurfaceKHR surface)
{
    int score = 0;
    VkPhysicalDeviceProperties props;
    VkPhysicalDeviceFeatures feas;
    vkGetPhysicalDeviceProperties(dev, &props);

    vkGetPhysicalDeviceFeatures(dev, &feas);

    if (IsDeviceSuitable(dev, surface) == false)
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
        /*for (int i = 0; i < glfwExtensionCount; i++)
        {
            std::cout << glfwExtensions[i] << std::endl;
        }*/
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
        int score = RateDeviceSutability(x, mSurface);
        //std::cout << "device " << i << " score " << score << std::endl;
        candidates.insert(std::make_pair(RateDeviceSutability(x, mSurface), x));
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
    QueueFamilyIndices indices = findQueueFamilies(mPhysicalDevice, mSurface);
    //std::cout << "physical score = " << RateDeviceSutability(mPhysicalDevice) << std::endl;
    //std::cout << "has value = " << indices.graphicsFamily.has_value() << std::endl;
    //exit(0);

    // 2. set the queue family info into the physical device
    std::vector<VkDeviceQueueCreateInfo> queueCreateinfos;
    std::set<uint32_t> uniqueQueueFamilies = {
        indices.graphicsFamily.value(), indices.presentFamily.value()};

    float queuePriority = 1.0f;
    for (uint32_t queueFamily : uniqueQueueFamilies)
    {
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = indices.graphicsFamily.value();
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateinfos.push_back(queueCreateInfo);
    }

    // 3.
    VkPhysicalDeviceFeatures deviceFeatures{};

    // 4. create logical device
    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pQueueCreateInfos = queueCreateinfos.data();
    createInfo.queueCreateInfoCount = queueCreateinfos.size();
    createInfo.pEnabledFeatures = &deviceFeatures;

    {
        // we don't need any more extensions here nowI
        {
            // set extensions count in the DeviceCreateInfo
            createInfo.enabledExtensionCount =
                static_cast<uint32_t>(deviceExtensions.size());
            createInfo.ppEnabledExtensionNames = deviceExtensions.data();
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

    // get the queue handles: graphics queue and presentQueue
    vkGetDeviceQueue(mDevice, indices.graphicsFamily.value(), 0, &mGraphicsQueue);
    vkGetDeviceQueue(mDevice, indices.presentFamily.value(), 0, &mPresentQueue);
}

/**
 * \brief       Create window surface for display
*/

void cDrawScene::CreateSurface()
{
    if (VK_SUCCESS != glfwCreateWindowSurface(mInstance, window, nullptr, &mSurface))
    {
        SIM_ERROR("glfw create surface for vulkan failed");
    }
}

/**
 * \brief       create the swap chain
*/
void cDrawScene::CreateSwapChain()
{
    //std::cout << "begin to create swap chain\n";
    // 1. fetch the details from physical device
    auto details = querySwapChainSupport(mPhysicalDevice, mSurface);
    // 2. set the format
    VkSurfaceFormatKHR format = chooseSwapSurfaceFormat(details.formats);
    // 3. set the present mode
    VkPresentModeKHR mode = chooseSwapPresentKHR(details.presentModes);
    // 4. set the extent (resolution of buffers)
    VkExtent2D extent = chooseSwapExtent(details.capabilities);

    // 5. set the size of swap chain
    uint32_t image_count = details.capabilities.minImageCount + 1;
    image_count = std::min(details.capabilities.maxImageCount, image_count);

    // 6. set up a big structure
    VkSwapchainCreateInfoKHR create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    create_info.surface = mSurface;

    create_info.minImageCount = image_count;
    create_info.imageFormat = format.format;
    create_info.imageColorSpace = format.colorSpace;
    create_info.imageExtent = extent;
    create_info.imageArrayLayers = 1;                             // always 1
    create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT; // we use swap chain image for directly rendering, namely color attachment
    create_info.presentMode = mode;

    /*
        6.1 image sharing mode
        exclusive: ownership changed when a image is visited by multiple queue
        concurrent: 
    */
    QueueFamilyIndices indices = findQueueFamilies(mPhysicalDevice, mSurface);
    uint32_t queueFamilyIndices[] = {indices.presentFamily.value(), indices.graphicsFamily.value()};
    if (indices.graphicsFamily == indices.presentFamily)
    {
        // the present, and graphics queue are the same, we can use the exclusive mode because the ownership can never be changed
        create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        create_info.queueFamilyIndexCount = 0;     // don't know
        create_info.pQueueFamilyIndices = nullptr; // don't know
    }
    else
    {
        // if they are different
        create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        create_info.queueFamilyIndexCount = 2;                // don't know
        create_info.pQueueFamilyIndices = queueFamilyIndices; // don't know
    }

    // 6.2 set the pretrasform (90 degree clockwise rotation,etc)
    create_info.preTransform = details.capabilities.currentTransform;

    create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR; // blend the color with other window
    create_info.clipped = VK_TRUE;                                  // window, color obsecured related...

    create_info.oldSwapchain = VK_NULL_HANDLE; // only one swapchain, so set it to false

    // 6.3 final create
    if (VK_SUCCESS != vkCreateSwapchainKHR(mDevice, &create_info, nullptr, &mSwapChain)) 
    {
        SIM_ERROR("create swap chain failed");
    }
    //std::cout << "succ to create swap chain\n";
    SIM_INFO("swapchain is created successfully");
}