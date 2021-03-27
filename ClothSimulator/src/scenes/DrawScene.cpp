#include "DrawScene.h"
#include "utils/LogUtil.h"
#include "utils/MathUtil.h"
#include "vulkan/vulkan.h"
#include <iostream>
#include <optional>
#include <set>
#ifdef _WIN32
#define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_INCLUDE_VULKAN
#include "GLFW/glfw3.h"
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#endif

#ifdef __linux__
#define VK_USE_PLATFORM_XCB_KHR
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_X11
#include <GLFW/glfw3native.h>
#endif

const std::vector<const char *> validationLayers = {
    "VK_LAYER_KHRONOS_validation"};
const std::vector<const char *> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME};
extern GLFWwindow *window;

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

const int MAX_FRAMES_IN_FLIGHT = 2;

#include "utils/MathUtil.h"
#include <array>
struct tVertex
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    tVector2f pos;
    tVector3f color;
    static VkVertexInputBindingDescription getBindingDescription()
    {
        VkVertexInputBindingDescription desc{};
        desc.binding = 0;
        desc.stride = sizeof(tVertex); // return the bytes of this type occupies
        desc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return desc;
    }

    /**
 * \brief       describe the description of the attributes. 
 *      
 *      we have two attributes, the position and color, we need to describe them individualy, use two strucutre.
 *      The first strucutre describe the first attribute (inPosition), we give the binding, location of this attribute, and the offset
 * 
 */
    static std::array<VkVertexInputAttributeDescription, 2>
    getAttributeDescriptions()
    {
        std::array<VkVertexInputAttributeDescription, 2> desc{};

        // position attribute
        desc[0].binding = 0; // binding point: 0
        desc[0].location = 0;
        desc[0].format = VK_FORMAT_R32G32_SFLOAT; // used for vec2f
        desc[0].offset = offsetof(tVertex, pos);

        // color attribute
        desc[1].binding = 0;
        desc[1].location = 1;
        desc[1].format = VK_FORMAT_R32G32B32_SFLOAT; // used for vec3f
        desc[1].offset = offsetof(tVertex, color);
        return desc;
    }
};

/**
 * \brief       manually point out the vertices info, include:
    1. position: vec2f in NDC
    2. color: vec3f \in [0, 1]
*/
const std::vector<tVertex> vertices = {{{0.0f, -0.5f}, {1.0f, 1.0f, 1.0f}},
                                       {{0.5f, 0.5f}, {1.0f, 0.0f, 1.0f}},
                                       {{-0.5f, 0.5f}, {1.0f, 1.0f, 0.0f}}};
cDrawScene::cDrawScene()
{
    mInstance = nullptr;
    mCurFrame = 0;
    mFrameBufferResized = false;
}

cDrawScene::~cDrawScene() { CleanVulkan(); }

void cDrawScene::CleanVulkan()
{
    CleanSwapChain();
    vkDestroyBuffer(mDevice, mVertexBuffer, nullptr);
    vkFreeMemory(mDevice, mVertexBufferMemory, nullptr);

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        vkDestroySemaphore(mDevice, mImageAvailableSemaphore[i], nullptr);
        vkDestroySemaphore(mDevice, mRenderFinishedSemaphore[i], nullptr);
        vkDestroyFence(mDevice, minFlightFences[i], nullptr);
    }
    vkDestroyCommandPool(mDevice, mCommandPool, nullptr);
    vkDestroyDevice(mDevice, nullptr);
    vkDestroySurfaceKHR(mInstance, mSurface, nullptr);
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
void cDrawScene::Update(double dt) { DrawFrame(); }

void cDrawScene::Resize(int w, int h) { mFrameBufferResized = true; }
/**
 * \brief           Reset the whole scene
*/
void cDrawScene::Reset() {}

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

    // 1. all changed
    CreateSwapChain();
    CreateImageViews();
    CreateRenderPass();
    CreateGraphicsPipeline();
    CreateFrameBuffers();
    CreateCommandPool();
    CreateVertexBuffer();
    CreateCommandBuffers();
    CreateSemaphores();
}

uint32_t findMemoryType(VkPhysicalDevice phy_device, uint32_t typeFilter,
                        VkMemoryPropertyFlags props)
{
    // get the memory info from the physical device
    VkPhysicalDeviceMemoryProperties mem_props{};
    vkGetPhysicalDeviceMemoryProperties(phy_device, &mem_props);

    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++)
    {
        // 1, 2, 4, 8, ...
        // the memory must meet the props (such as visible from the CPU)
        if ((typeFilter & (1 << i)) &&
            (mem_props.memoryTypes[i].propertyFlags & props) == props)
        {
            return i;
        }
    }
    SIM_ERROR("failed to find suitable memory type for filter {} in {} types",
              typeFilter, mem_props.memoryTypeCount);
    return 0;
}

void cDrawScene::CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                              VkMemoryPropertyFlags props, VkBuffer &buffer,
                              VkDeviceMemory &buffer_memory)
{
    // 1. create a vertex buffer
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    // buffer_info.size =
    //     sizeof(vertices[0]) *
    //     vertices
    //         .size(); // do not use sizeof(vertices) because it will be a little bigger than what it needs (dynamic vector)
    buffer_info.size = size;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    SIM_ASSERT(vkCreateBuffer(mDevice, &buffer_info, nullptr, &buffer) ==
               VK_SUCCESS);

    // 2. allocate: first meet the memory requirements
    VkMemoryRequirements mem_reqs{};
    vkGetBufferMemoryRequirements(mDevice, buffer, &mem_reqs);
    // mem_reqs
    //     .size; // the size of required amount of memorys in bytes, may differ from the "bufferInfo.size"
    // mem_reqs.alignment;      // the beginning address of this buffer
    // mem_reqs.memoryTypeBits; // unknown

    // 3. allocate: memory allocation
    VkMemoryAllocateInfo allo_info{};
    allo_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allo_info.allocationSize = mem_reqs.size;
    allo_info.memoryTypeIndex =
        findMemoryType(mPhysicalDevice, mem_reqs.memoryTypeBits, props);

    SIM_ASSERT(vkAllocateMemory(mDevice, &allo_info, nullptr, &buffer_memory) ==
               VK_SUCCESS);

    // 4. bind (connect) the buffer with the allocated memory
    vkBindBufferMemory(mDevice, buffer, buffer_memory, 0);
}

void cDrawScene::CopyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer,
                            VkDeviceSize size)
{
    // 1. create a command buffer
    VkCommandBufferAllocateInfo allo_info{};
    allo_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allo_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allo_info.commandPool = mCommandPool;
    allo_info.commandBufferCount = 1;
    VkCommandBuffer cmd_buffer;
    vkAllocateCommandBuffers(mDevice, &allo_info, &cmd_buffer);

    // 2. begin to record the command buffer
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags =
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; // we only use this command buffer for a single time
    vkBeginCommandBuffer(cmd_buffer, &begin_info);

    // 3. copy from src to dst buffer
    VkBufferCopy copy_region{};
    copy_region.srcOffset = 0;
    copy_region.dstOffset = 0;
    copy_region.size = size;
    vkCmdCopyBuffer(cmd_buffer, srcBuffer, dstBuffer, 1, &copy_region);

    // 4. end recording
    vkEndCommandBuffer(cmd_buffer);

    // 5. submit info
    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd_buffer;

    vkQueueSubmit(mGraphicsQueue, 1, &submit_info, VK_NULL_HANDLE);

    // wait, till the queue is empty (which means all commands have been finished)
    vkQueueWaitIdle(mGraphicsQueue);

    // 6. deconstruct
    vkFreeCommandBuffers(mDevice, mCommandPool, 1, &cmd_buffer);
}

void cDrawScene::CreateVertexBuffer()
{
    VkDeviceSize buffer_size = sizeof(vertices[0]) * vertices.size();
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    CreateBuffer(buffer_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 stagingBuffer, stagingBufferMemory);

    // 5. copy the vertex data to the buffer
    void *data = nullptr;
    // map the memory to "data" ptr;
    vkMapMemory(mDevice, stagingBufferMemory, 0, buffer_size, 0, &data);

    // write the data
    memcpy(data, vertices.data(), buffer_size);

    // unmap
    vkUnmapMemory(mDevice, stagingBufferMemory);

    CreateBuffer(buffer_size,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, mVertexBuffer,
                 mVertexBufferMemory);

    CopyBuffer(stagingBuffer, mVertexBuffer, buffer_size);
    vkDestroyBuffer(mDevice, stagingBuffer, nullptr);
    vkFreeMemory(mDevice, stagingBufferMemory, nullptr);
}

/**
 * \brief           draw a single frame
        1. get an image from the swap chain
        2. executate the command buffer with that image as attachment in the framebuffer
        3. return the image to the swap chain for presentation
    These 3 actions, ideally should be executed asynchronously. but the function calls will return before the operations are actually finished.
    So the order of execution is undefined. we need "fences" ans "semaphores" to synchronizing swap chain

*/
void cDrawScene::DrawFrame()
{
    vkWaitForFences(mDevice, 1, &minFlightFences[mCurFrame], VK_TRUE,
                    UINT64_MAX);

    // 1. acquire an image from the swap chain
    uint32_t imageIndex;
    VkResult result = vkAcquireNextImageKHR(mDevice, mSwapChain, UINT64_MAX,
                                            mImageAvailableSemaphore[mCurFrame],
                                            VK_NULL_HANDLE, &imageIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || mFrameBufferResized == true)
    {
        // the window may be resized, we need to recreate it
        mFrameBufferResized = false;
        RecreateSwapChain();
        return;
    }
    else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
    {
        SIM_ERROR("error");
    }

    if (mImagesInFlight[imageIndex] != VK_NULL_HANDLE)
    {
        vkWaitForFences(mDevice, 1, &mImagesInFlight[imageIndex], VK_TRUE,
                        UINT64_MAX);
    }
    mImagesInFlight[imageIndex] = mImagesInFlight[mCurFrame];

    // 2. submitting the command buffer
    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    VkSemaphore waitSemaphores[] = {mImageAvailableSemaphore[mCurFrame]};
    VkPipelineStageFlags waitStages[] = {
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT}; //
    submit_info.waitSemaphoreCount =
        1; // how much semaphres does this submission need to wait?
    submit_info.waitSemaphoreCount = 1;
    submit_info.pWaitSemaphores = waitSemaphores;
    submit_info.pWaitDstStageMask = waitStages;

    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &mCommandBuffers[imageIndex];

    // when the commands are finished, which semaphore do we need to send?
    VkSemaphore signalSemaphores[] = {mRenderFinishedSemaphore[mCurFrame]};
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = signalSemaphores;

    vkResetFences(mDevice, 1, &minFlightFences[mCurFrame]);
    if (vkQueueSubmit(mGraphicsQueue, 1, &submit_info,
                      minFlightFences[mCurFrame]) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to submit draw command buffer!");
        exit(0);
    }

    // 3. render finish and present the image to the screen
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;
    VkSwapchainKHR swapChains[] = {mSwapChain};
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &imageIndex;
    vkQueuePresentKHR(mPresentQueue, &presentInfo);

    // we wait the GPU to finish its work after submitting
    vkQueueWaitIdle(mPresentQueue);
    mCurFrame = (mCurFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

// which queues do we want to support?
struct QueueFamilyIndices
{
    std::optional<uint32_t>
        graphicsFamily; // here we use optional, because any unit value would be valid and we need to distinguish from non-value case
    std::optional<uint32_t>
        presentFamily; // the ability to show the image to the screen
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
QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device,
                                     VkSurfaceKHR surface)
{
    QueueFamilyIndices indices;
    uint32_t num_of_families = 0;

    // 1. get all queue faimilies on this physical device
    vkGetPhysicalDeviceQueueFamilyProperties(device, &num_of_families, nullptr);
    std::vector<VkQueueFamilyProperties> families(
        num_of_families); // storage the properties of each queue on this physical device. such as maximium number of queues, types of queues...

    vkGetPhysicalDeviceQueueFamilyProperties(device, &num_of_families,
                                             families.data());

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
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface,
                                             &presentSupport);
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
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                         nullptr);
    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                         availableExtensions.data());

    std::set<std::string> requiredExtensions(deviceExtensions.begin(),
                                             deviceExtensions.end());
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
SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device,
                                              VkSurfaceKHR surface)
{
    SwapChainSupportDetails details;

    // 1. capabilities
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface,
                                              &details.capabilities);

    // 2. formats
    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount,
                                         nullptr);
    details.formats.resize(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount,
                                         details.formats.data());

    // 3. presentModes
    uint32_t modesCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &modesCount,
                                              nullptr);

    details.presentModes.resize(modesCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &modesCount,
                                              details.presentModes.data());

    return details;
}

/**
 * \brief       select the best surface format(color channels and types)
*/
VkSurfaceFormatKHR
chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR> availableFormats)
{
    for (auto &x : availableFormats)
    {
        if (x.format == VK_FORMAT_B8G8R8A8_SRGB &&
            x.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            return x;
    }
    return availableFormats[0];
}

/**
 * \brief       choose the swapchain present mode
 * 
 *          the author thought the "mailbox" which can support the triple buffering is the best option
*/
VkPresentModeKHR
chooseSwapPresentKHR(const std::vector<VkPresentModeKHR> &modes)
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

        VkExtent2D actualExtent = {static_cast<uint32_t>(width),
                                   static_cast<uint32_t>(height)};

        actualExtent.width = std::max(
            capabilities.minImageExtent.width,
            std::min(capabilities.maxImageExtent.width, actualExtent.width));
        actualExtent.height = std::max(
            capabilities.minImageExtent.height,
            std::min(capabilities.maxImageExtent.height, actualExtent.height));

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
        swapChainAdequate = details.formats.empty() == false &&
                            (details.presentModes.empty()) == false;
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
    if (enableValidationLayers == true &&
        CheckValidationLayerSupport() == false)
    {
        SIM_ERROR("Validation Layers is not supported");
    }
    // 2. application info
    VkApplicationInfo
        appInfo{}; // it contains a pNext member for further extension
    appInfo.sType =
        VK_STRUCTURE_TYPE_APPLICATION_INFO; // type should be set manually
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
    SIM_ASSERT(vkCreateInstance(&createInfo, nullptr, &mInstance) ==
               VK_SUCCESS);
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
void cDrawScene::SetupDebugMessenger() {}

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
    std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(),
                                              indices.presentFamily.value()};

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

    if (vkCreateDevice(mPhysicalDevice, &createInfo, nullptr, &mDevice) !=
        VK_SUCCESS)
    {
        SIM_ERROR("create logic device failed");
    }

    // get the queue handles: graphics queue and presentQueue
    vkGetDeviceQueue(mDevice, indices.graphicsFamily.value(), 0,
                     &mGraphicsQueue);
    vkGetDeviceQueue(mDevice, indices.presentFamily.value(), 0, &mPresentQueue);
}

/**
 * \brief       Create window surface for display
*/

void cDrawScene::CreateSurface()
{
    if (VK_SUCCESS !=
        glfwCreateWindowSurface(mInstance, window, nullptr, &mSurface))
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
    create_info.imageArrayLayers = 1; // always 1
    create_info.imageUsage =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT; // we use swap chain image for directly rendering, namely color attachment
    create_info.presentMode = mode;

    /*
        6.1 image sharing mode
        exclusive: ownership changed when a image is visited by multiple queue
        concurrent:
    */
    QueueFamilyIndices indices = findQueueFamilies(mPhysicalDevice, mSurface);
    uint32_t queueFamilyIndices[] = {indices.presentFamily.value(),
                                     indices.graphicsFamily.value()};
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

    create_info.compositeAlpha =
        VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR; // blend the color with other window
    create_info.clipped = VK_TRUE;         // window, color obsecured related...

    create_info.oldSwapchain =
        VK_NULL_HANDLE; // only one swapchain, so set it to false

    // 6.3 final create
    if (VK_SUCCESS !=
        vkCreateSwapchainKHR(mDevice, &create_info, nullptr, &mSwapChain))
    {
        SIM_ERROR("create swap chain failed");
    }
    //std::cout << "succ to create swap chain\n";
    uint32_t swapchain_image_count = 0;
    vkGetSwapchainImagesKHR(mDevice, mSwapChain, &swapchain_image_count,
                            nullptr);
    mSwapChainImages.resize(swapchain_image_count);
    vkGetSwapchainImagesKHR(mDevice, mSwapChain, &swapchain_image_count,
                            mSwapChainImages.data());
    mSwapChainImageFormat = format.format;
    mSwapChainExtent = extent;
    SIM_INFO("swapchain is created successfully");
}

/**
 * \brief           create "view" for swap chain images
 * 
 *          Frankly, we need to know how the SwapChainImages is accessed
 * */
void cDrawScene::CreateImageViews()
{
    mSwapChainImageViews.resize(mSwapChainImages.size());
    for (int i = 0; i < mSwapChainImageViews.size(); i++)
    {
        VkImageViewCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        createInfo.image = mSwapChainImages
            [i]; // which image will this imageview to be binded to
        createInfo.viewType =
            VK_IMAGE_VIEW_TYPE_2D; // 1d/2d/3d textures, cube maps
        createInfo.format = this->mSwapChainImageFormat;
        // map all channels to red channels, for example
        createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        // visiting range
        createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;
        if (VK_SUCCESS != vkCreateImageView(mDevice, &createInfo, nullptr,
                                            &(mSwapChainImageViews[i])))
        {
            SIM_ERROR("create image view {} failed", i);
        }
    }
}

/**
 * \brief               Create Graphcis Pipeline
*/
#include "utils/FileUtil.h"
std::vector<char> ReadFile(const std::string &filename)
{
    SIM_ASSERT(cFileUtil::ExistsFile(filename) == true);
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    SIM_INFO("read file {} size {} succ", filename, fileSize);
    return buffer;
}

VkShaderModule cDrawScene::CreateShaderModule(const std::vector<char> &code)
{
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t *>(code.data());
    VkShaderModule module;
    // SIM_ASSERT(VK_SUCCESS ==);
    vkCreateShaderModule(mDevice, &createInfo, nullptr, &module);
    return module;
}

void cDrawScene::CreateGraphicsPipeline()
{
    // load and create the module
    auto VertShaderCode = ReadFile("src/shaders/shader.vert.spv");
    auto FragShaderCode = ReadFile("src/shaders/shader.frag.spv");
    VkShaderModule VertShaderModule = CreateShaderModule(VertShaderCode);
    VkShaderModule FragShaderModule = CreateShaderModule(FragShaderCode);

    // put the programmable shaders into the pipeline
    // {
    VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.module = VertShaderModule;
    vertShaderStageInfo.pName = "main"; // entry point
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.pSpecializationInfo =
        nullptr; // we can define some constants for optimizatin here

    VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.module = FragShaderModule;
    fragShaderStageInfo.pName = "main";
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.pSpecializationInfo = nullptr;

    VkPipelineShaderStageCreateInfo shader_stages[] = {vertShaderStageInfo,
                                                       fragShaderStageInfo};
    // }

    // put the fixed stages into the pipeline
    // {
    auto bindingDesc = tVertex::getBindingDescription();
    auto attriDesc = tVertex::getAttributeDescriptions();
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDesc;
    vertexInputInfo.vertexAttributeDescriptionCount =
        static_cast<uint32_t>(attriDesc.size());
    vertexInputInfo.pVertexAttributeDescriptions = attriDesc.data();

    // don't know what's this
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType =
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.primitiveRestartEnable = VK_FALSE;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    // Viewport: which part of framebuffer will be rendered to
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float)mSwapChainExtent.width;
    viewport.height = (float)mSwapChainExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    // scissors: useless
    VkRect2D scissor{};
    scissor.extent = mSwapChainExtent;
    scissor.offset = {0, 0};

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.pScissors = &scissor;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.viewportCount = 1;

    // rasterizer: vertices into fragments
    VkPipelineRasterizationStateCreateInfo raster_info{};
    raster_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster_info.depthClampEnable =
        VK_FALSE; // clamp the data outside of the near-far plane insteand of deleting them
    raster_info.rasterizerDiscardEnable =
        VK_FALSE; // disable the rasterization, it certainly should be disable
    raster_info.polygonMode = VK_POLYGON_MODE_FILL; // normal
    raster_info.lineWidth =
        1.0f; // if not 1.0, we need to enable the GPU "line_width" feature
    raster_info.cullMode = VK_CULL_MODE_BACK_BIT; // back cull
    raster_info.frontFace =
        VK_FRONT_FACE_CLOCKWISE; // define the vertex order for front-facing

    // setting for possible shadow map
    raster_info.depthBiasEnable = VK_FALSE;
    raster_info.depthBiasConstantFactor = 0.0f;
    raster_info.depthBiasClamp = 0.0f;
    raster_info.depthBiasSlopeFactor = 0.0f;

    // setting for multisampling
    VkPipelineMultisampleStateCreateInfo multisampling_info{};
    multisampling_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling_info.sampleShadingEnable = VK_FALSE;
    multisampling_info.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampling_info.minSampleShading = 1.0f;
    multisampling_info.pSampleMask = nullptr;
    multisampling_info.alphaToCoverageEnable = VK_FALSE;
    multisampling_info.alphaToOneEnable = VK_FALSE;

    // setting for color blending: the interaction between framebuffer and the output of fragment shader
    // configuration for each individual frame buffer
    VkPipelineColorBlendAttachmentState colorBlendAttachmentState{};
    // which channels will be effected?
    colorBlendAttachmentState.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachmentState.blendEnable = VK_FALSE;

    // the global color blending setting
    VkPipelineColorBlendStateCreateInfo colorBlendState_info{};
    colorBlendState_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlendState_info.logicOpEnable = VK_FALSE;
    colorBlendState_info.logicOp = VK_LOGIC_OP_COPY;
    colorBlendState_info.attachmentCount = 1;
    colorBlendState_info.pAttachments = &colorBlendAttachmentState;
    colorBlendState_info.blendConstants[0] = 0.0f;
    colorBlendState_info.blendConstants[1] = 0.0f;
    colorBlendState_info.blendConstants[2] = 0.0f;
    colorBlendState_info.blendConstants[3] = 0.0f;

    // some settings can be dynamically changed without recreating the pipelines
    // VkDynamicState dynamicStates[] = {
    // VK_DYNAMIC_STATE_VIEWPORT,
    // VK_DYNAMIC_STATE_LINE_WIDTH
    // };

    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    // dynamicState.dynamicStateCount = 2;
    dynamicState.dynamicStateCount = 0;
    // dynamicState.pDynamicStates = dynamicStates;
    dynamicState.pDynamicStates = nullptr;

    // pipeline layout (uniform values in our shaders)
    // now here is an empty pipeline layout
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 0;
    pipelineLayoutInfo.pSetLayouts = nullptr;
    pipelineLayoutInfo.pushConstantRangeCount = 0;
    pipelineLayoutInfo.pPushConstantRanges = nullptr;

    SIM_ASSERT(vkCreatePipelineLayout(mDevice, &pipelineLayoutInfo, nullptr,
                                      &mPipelineLayout) == VK_SUCCESS);
    // }

    // create the final pipeline
    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shader_stages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &raster_info;
    pipelineInfo.pMultisampleState = &multisampling_info;
    pipelineInfo.pDepthStencilState = nullptr;
    pipelineInfo.pColorBlendState = &colorBlendState_info;
    pipelineInfo.pDynamicState = nullptr;
    pipelineInfo.layout = mPipelineLayout;
    pipelineInfo.renderPass = mRenderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineInfo.basePipelineIndex = -1;

    SIM_ASSERT(vkCreateGraphicsPipelines(mDevice, VK_NULL_HANDLE, 1,
                                         &pipelineInfo, nullptr,
                                         &mGraphicsPipeline) == VK_SUCCESS)

    // destory the modules
    vkDestroyShaderModule(mDevice, VertShaderModule, nullptr);
    vkDestroyShaderModule(mDevice, FragShaderModule, nullptr);

    SIM_INFO("Create graphics pipeline succ");
}

void cDrawScene::CreateRenderPass()
{
    // describe the renderpass setting
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = mSwapChainImageFormat;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT; // no multisampling
    colorAttachment.loadOp =
        VK_ATTACHMENT_LOAD_OP_CLEAR; // what to do with the data in the attachement before rendering: preserve, clear, don't care
    colorAttachment.storeOp =
        VK_ATTACHMENT_STORE_OP_STORE; // what to do ... after rendering: preserve, don't care
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;   //
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE; //
    colorAttachment.initialLayout =
        VK_IMAGE_LAYOUT_UNDEFINED; // which layout will the image have, before render pass
    colorAttachment.finalLayout =
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; // which ... after render pass

    // attachment renference for attachment description, used for differnet subpasses
    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment =
        0; // we have only one attachement, so it's 0
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    // subpass
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint =
        VK_PIPELINE_BIND_POINT_GRAPHICS; // for display, not computing
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    // render pass
    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &colorAttachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;
    SIM_ASSERT(vkCreateRenderPass(mDevice, &renderPassInfo, nullptr,
                                  &mRenderPass) == VK_SUCCESS);
}

void cDrawScene::CreateFrameBuffers()
{
    // framebuffers size = image views size
    mSwapChainFramebuffers.resize(mSwapChainImageViews.size());

    for (int i = 0; i < mSwapChainImageViews.size(); i++)
    {
        VkImageView attachments[] = {mSwapChainImageViews[i]};
        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = mRenderPass;
        framebufferInfo.attachmentCount = 1;
        framebufferInfo.pAttachments = attachments;
        framebufferInfo.width = mSwapChainExtent.width;
        framebufferInfo.height = mSwapChainExtent.height;
        framebufferInfo.layers = 1;
        SIM_ASSERT(VK_SUCCESS ==
                   vkCreateFramebuffer(mDevice, &framebufferInfo, nullptr,
                                       &(mSwapChainFramebuffers[i])));
        // SIM_ASSERT;
    }
}

void cDrawScene::CreateCommandPool()
{
    // VkCommandPool mCommandPool;
    auto queue_families = findQueueFamilies(mPhysicalDevice, mSurface);
    VkCommandPoolCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    info.queueFamilyIndex = queue_families.graphicsFamily.value();
    SIM_ASSERT(vkCreateCommandPool(mDevice, &info, nullptr, &mCommandPool) ==
               VK_SUCCESS);
    SIM_INFO("Create Command Pool succ");
}

/**
 * \brief           Create and fill the command buffer
*/
void cDrawScene::CreateCommandBuffers()
{
    // 1. create the command buffers
    mCommandBuffers.resize(mSwapChainFramebuffers.size());
    VkCommandBufferAllocateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    info.commandPool = mCommandPool;
    info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    info.commandBufferCount = (uint32_t)mCommandBuffers.size();
    SIM_ASSERT(vkAllocateCommandBuffers(mDevice, &info,
                                        mCommandBuffers.data()) == VK_SUCCESS);

    // 2. record the command buffers
    for (int i = 0; i < mCommandBuffers.size(); i++)
    {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.pInheritanceInfo = nullptr;
        beginInfo.flags = 0;
        SIM_ASSERT(VK_SUCCESS ==
                   vkBeginCommandBuffer(mCommandBuffers[i], &beginInfo));

        // 3. start a render pass
        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = mRenderPass;
        renderPassInfo.framebuffer = mSwapChainFramebuffers[i];

        renderPassInfo.renderArea.extent = mSwapChainExtent;
        renderPassInfo.renderArea.offset = {0, 0};

        // full black clear color
        VkClearValue clear_color = {0.0f, 0.0f, 0.0f, 1.0f};
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clear_color;

        // begin render pass
        vkCmdBeginRenderPass(mCommandBuffers[i], &renderPassInfo,
                             VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(mCommandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
                          mGraphicsPipeline);

        VkBuffer vertexBuffers[] = {mVertexBuffer};
        VkDeviceSize offsets[] = {0};
        vkCmdBindVertexBuffers(mCommandBuffers[0], 0, 1, vertexBuffers,
                               offsets);

        // draaaaaaaaaaaaaaaaaaaaaaaaaaaaw!
        vkCmdDraw(mCommandBuffers[i], 3, 1, 0, 0);

        // end render pass
        vkCmdEndRenderPass(mCommandBuffers[i]);

        SIM_ASSERT(VK_SUCCESS == vkEndCommandBuffer(mCommandBuffers[i]));
    }
    SIM_INFO("Create Command buffers succ");
}

/**
 * \brief           Create (two) semaphores
*/
void cDrawScene::CreateSemaphores()
{
    mImageAvailableSemaphore.resize(MAX_FRAMES_IN_FLIGHT);
    mRenderFinishedSemaphore.resize(MAX_FRAMES_IN_FLIGHT);
    minFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
    mImagesInFlight.resize(mSwapChainImages.size(), VK_NULL_HANDLE);

    VkSemaphoreCreateInfo semaphore_info{};
    semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        SIM_ASSERT(
            (VK_SUCCESS == vkCreateSemaphore(mDevice, &semaphore_info, nullptr,
                                             &mImageAvailableSemaphore[i])) &&
            (VK_SUCCESS == vkCreateSemaphore(mDevice, &semaphore_info, nullptr,
                                             &mRenderFinishedSemaphore[i])) &&
            (vkCreateFence(mDevice, &fence_info, nullptr,
                           &minFlightFences[i]) == VK_SUCCESS));
    }
    SIM_INFO("Create semaphores succ");
}

/**
 * \brief           recreate the swap chain when the window is resized
*/
void cDrawScene::RecreateSwapChain()
{
    // wait
    vkDeviceWaitIdle(mDevice);

    // clean
    CleanSwapChain();

    // recreate
    CreateSwapChain();
    CreateImageViews();
    CreateRenderPass();
    CreateGraphicsPipeline();
    CreateFrameBuffers();
    CreateCommandBuffers();
}

void cDrawScene::CleanSwapChain()
{
    for (auto &x : mSwapChainFramebuffers)
        vkDestroyFramebuffer(mDevice, x, nullptr);
    vkFreeCommandBuffers(mDevice, mCommandPool,
                         static_cast<uint32_t>(mCommandBuffers.size()),
                         mCommandBuffers.data());
    vkDestroyPipeline(mDevice, mGraphicsPipeline, nullptr);
    vkDestroyPipelineLayout(mDevice, mPipelineLayout, nullptr);
    vkDestroyRenderPass(mDevice, mRenderPass, nullptr);
    for (auto &x : mSwapChainImageViews)
    {
        vkDestroyImageView(mDevice, x, nullptr);
    }
    vkDestroySwapchainKHR(mDevice, mSwapChain, nullptr);
}
