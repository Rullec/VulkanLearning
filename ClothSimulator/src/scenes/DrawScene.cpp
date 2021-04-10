#include "DrawScene.h"
#include "SceneBuilder.h"
#include "cameras/ArcBallCamera.h"
#include "geometries/Primitives.h"
#include "glm/glm.hpp"
#include "scenes/SimScene.h"
#include "utils/JsonUtil.h"
#include "utils/LogUtil.h"
#include "vulkan/vulkan.h"
#include <iostream>
#include <optional>
#include <set>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLFW_INCLUDE_VULKAN
#ifdef _WIN32
#define VK_USE_PLATFORM_WIN32_KHR
#include "GLFW/glfw3.h"
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#endif

#ifdef __linux__
#define VK_USE_PLATFORM_XCB_KHR
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_X11
#include <GLFW/glfw3native.h>
#endif

#ifdef __APPLE__
#include <GLFW/glfw3.h>
#endif

std::vector<const char *> validationLayers = {"VK_LAYER_KHRONOS_validation"};

#ifdef __APPLE__
std::vector<const char *> deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME,
                                              "VK_KHR_portability_subset"};
#else
std::vector<const char *> deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
#endif

extern GLFWwindow *window;

#ifdef NDEBUG
bool enableValidationLayers = false;
#else
bool enableValidationLayers = true;
#endif

const int MAX_FRAMES_IN_FLIGHT = 2;

float fov = 45.0f;
float near = 0.1f;
float far = 100.0f;

#include "utils/MathUtil.h"
#include <array>
VkVertexInputBindingDescription tVkVertex::getBindingDescription()
{
    VkVertexInputBindingDescription desc{};
    desc.binding = 0;
    desc.stride = sizeof(tVkVertex); // return the bytes of this type occupies
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
std::array<VkVertexInputAttributeDescription, 3>
tVkVertex::getAttributeDescriptions()
{
    std::array<VkVertexInputAttributeDescription, 3> desc{};

    // position attribute
    desc[0].binding = 0; // binding point: 0
    desc[0].location = 0;
    desc[0].format = VK_FORMAT_R32G32B32_SFLOAT; // used for vec3f
    // desc[0].format = VK_FORMAT_R32G32_SFLOAT; // used for vec2f
    desc[0].offset = offsetof(tVkVertex, pos);

    // color attribute
    desc[1].binding = 0;
    desc[1].location = 1;
    desc[1].format = VK_FORMAT_R32G32B32_SFLOAT; // used for vec3f
    desc[1].offset = offsetof(tVkVertex, color);

    // texture atrribute
    desc[2].binding = 0;
    desc[2].location = 2;
    desc[2].format = VK_FORMAT_R32G32_SFLOAT; // used for vec2f
    desc[2].offset = offsetof(tVkVertex, texCoord);

    return desc;
}

/**
 * \brief       manually point out the vertices info, include:
    1. position: vec2f in NDC
    2. color: vec3f \in [0, 1]
*/
const float ground_scale = 100.0;
std::vector<tVkVertex> ground_vertices = {
    {{50.0f, 0.0f, -50.0f}, {0.7f, 0.7f, 0.7f}, {ground_scale, 0.0f}},
    {{-50.0f, 0.0f, -50.0f}, {0.7f, 0.7f, 0.7f}, {0.0f, 0.0f}},
    {{-50.0f, 0.0f, 50.0f}, {0.7f, 0.7f, 0.7f}, {0.0f, ground_scale}},

    {{50.0f, 0.0f, -50.0f}, {0.7f, 0.7f, 0.7f}, {ground_scale, 0.0f}},
    {{-50.0f, 0.0f, 50.0f}, {0.7f, 0.7f, 0.7f}, {0.0f, ground_scale}},
    {{50.0f, 0.0f, 50.0f}, {0.7f, 0.7f, 0.7f}, {ground_scale, ground_scale}},
};

tVector cDrawScene::GetCameraPos() const
{
    tVector pos = tVector::Ones();
    pos.segment(0, 3) = mCamera->pos.cast<double>();
    return pos;
}
bool cDrawScene::IsMouseRightButton(int glfw_button)
{
    return glfw_button == GLFW_MOUSE_BUTTON_RIGHT;
}

bool cDrawScene::IsRelease(int glfw_action)
{
    return GLFW_RELEASE == glfw_action;
}

bool cDrawScene::IsPress(int glfw_action) { return GLFW_PRESS == glfw_action; }

tVector cDrawScene::CalcCursorPointWorldPos() const
{
    tMatrix mat;
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);
    // printf("[debug] cursor xpos %.3f, ypos %.3f\n", xpos, ypos);
    int height = mSwapChainExtent.height, width = mSwapChainExtent.width;
    // shape the conversion mat
    tVector test = tVector(xpos, ypos, 1, 1);
    tMatrix mat1 = tMatrix::Identity();
    mat1(0, 0) = 1.0 / width;
    mat1(0, 3) = 0.5 / width;
    mat1(1, 1) = 1.0 / height;
    mat1(1, 3) = 0.5 / height;
    // std::cout << "after 1, vec = "
    //           << (test = mat1 * test).transpose() << std::endl;

    tMatrix mat2 = tMatrix::Identity();
    mat2(0, 0) = 2;
    mat2(0, 3) = -1;
    mat2(1, 1) = -2;
    mat2(1, 3) = 1;
    // std::cout << "after 2, vec = "
    //           << (test = mat2 * test).transpose() << std::endl;

    // pos = mat2 * pos;
    tMatrix mat3 = tMatrix::Identity();
    // mat3(0, 0) = std::tan(cMathUtil::Radians(mFov) / 2) * mNear;
    mat3(0, 0) = width * 1.0 / height * std::tan(glm::radians(fov) / 2) * near;
    mat3(1, 1) = std::tan(glm::radians(fov) / 2) * near;
    mat3(2, 2) = 0, mat3(2, 3) = -near;
    // std::cout << "after 3, vec = "
    //           << (test = mat3 * test).transpose() << std::endl;

    // std::cout << "mat 3 = " << mat3 << std::endl;
    // exit(1);
    // pos = mat3 * pos;
    tMatrix mat4 = mCamera->ViewMatrix().inverse().cast<double>();
    // std::cout << "after 4, vec = "
    //           << (test = mat4 * test).transpose() << std::endl;
    // std::cout <<"dir = " <<  (test - mCamera->GetCameraPos()).normalized().transpose() << std::endl;
    mat = mat4 * mat3 * mat2 * mat1;
    // exit(1);

    tVector pos = mat * tVector(xpos, ypos, 1, 1);
    // tVector camera_pos = tVector::Ones();
    // camera_pos.segment(0, 3) = mCamera->pos.cast<double>();
    return pos;
    // tRay *ray = new tRay(camera_pos, pos);
    // std::cout << "ray origin " << ray->mOrigin.transpose()
    //           << ", target = " << pos.transpose() << std::endl;
    // mSimScene->RayCast(ray);
}

// };
cDrawScene::cDrawScene()
{
    mInstance = nullptr;
    mCurFrame = 0;
    mFrameBufferResized = false;
    mLeftButtonPress = false;
}

cDrawScene::~cDrawScene() {}

void cDrawScene::CleanVulkan()
{
    CleanSwapChain();

    vkDestroySampler(mDevice, mTextureSampler, nullptr);
    vkDestroyImageView(mDevice, mTextureImageView, nullptr);
    vkDestroyImage(mDevice, mTextureImage, nullptr);
    vkFreeMemory(mDevice, mTextureImageMemory, nullptr);

    vkDestroyImageView(mDevice, mDepthImageView, nullptr);
    vkDestroyImage(mDevice, mDepthImage, nullptr);
    vkFreeMemory(mDevice, mDepthImageMemory, nullptr);

    vkDestroyDescriptorSetLayout(mDevice, mDescriptorSetLayout, nullptr);
    vkDestroyBuffer(mDevice, mVertexBufferCloth, nullptr);
    vkFreeMemory(mDevice, mVertexBufferMemoryCloth, nullptr);
    vkDestroyBuffer(mDevice, mVertexBufferGround, nullptr);
    vkFreeMemory(mDevice, mVertexBufferMemoryGround, nullptr);

    vkDestroyBuffer(mDevice, mLineBuffer, nullptr);
    vkFreeMemory(mDevice, mLineBufferMemory, nullptr);

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

void cDrawScene::CreateGraphicsPipeline(const std::string mode,
                                        VkPipeline &pipeline)
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
    auto bindingDesc = tVkVertex::getBindingDescription();
    auto attriDesc = tVkVertex::getAttributeDescriptions();
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

    if (mode == "triangle")
    {
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    }
    else if (mode == "line")
    {
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    }
    else
    {
        SIM_ERROR("unsupported mode {}", mode);
        exit(0);
    }

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

    // raster_info.cullMode = VK_CULL_MODE_NONE; // don't cull
    raster_info.cullMode = VK_CULL_MODE_BACK_BIT; // back cull
    raster_info.frontFace =
        VK_FRONT_FACE_COUNTER_CLOCKWISE; // define the vertex order for front-facing

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
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &mDescriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 0;
    pipelineLayoutInfo.pPushConstantRanges = nullptr;

    SIM_ASSERT(vkCreatePipelineLayout(mDevice, &pipelineLayoutInfo, nullptr,
                                      &mPipelineLayout) == VK_SUCCESS);
    // }

    // add depth test
    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType =
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.minDepthBounds = 0.0f; // Optional
    depthStencil.maxDepthBounds = 1.0f; // Optional
    depthStencil.stencilTestEnable = VK_FALSE;
    depthStencil.front = {}; // Optional
    depthStencil.back = {};  // Optional

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
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlendState_info;
    pipelineInfo.pDynamicState = nullptr;
    pipelineInfo.layout = mPipelineLayout;
    pipelineInfo.renderPass = mRenderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineInfo.basePipelineIndex = -1;
    SIM_ASSERT(vkCreateGraphicsPipelines(mDevice, VK_NULL_HANDLE, 1,
                                         &pipelineInfo, nullptr,
                                         &pipeline) == VK_SUCCESS)

    // destory the modules
    vkDestroyShaderModule(mDevice, VertShaderModule, nullptr);
    vkDestroyShaderModule(mDevice, FragShaderModule, nullptr);

    SIM_INFO("Create graphics pipeline succ");
}

/**
 * \brief       Init vulkan and other stuff
*/
void cDrawScene::Init(const std::string &conf_path)
{
    // init camera pos

    {
        Json::Value root;
        cJsonUtil::LoadJson(conf_path, root);
        Json::Value camera_json = cJsonUtil::ParseAsValue("camera", root);
        Json::Value camera_pos_json =
            cJsonUtil::ParseAsValue("camera_pos", camera_json);
        Json::Value camera_focus_json =
            cJsonUtil::ParseAsValue("camera_focus", camera_json);
        SIM_ASSERT(camera_pos_json.size() == 3);
        SIM_ASSERT(camera_focus_json.size() == 3);
        mCameraInitFocus = tVector3f(camera_focus_json[0].asFloat(),
                                     camera_focus_json[1].asFloat(),
                                     camera_focus_json[2].asFloat());
        mCameraInitPos = tVector3f(camera_pos_json[0].asFloat(),
                                   camera_pos_json[1].asFloat(),
                                   camera_pos_json[2].asFloat());
        fov = cJsonUtil::ParseAsFloat("fov", camera_json);
        near = cJsonUtil::ParseAsFloat("near", camera_json);
        far = cJsonUtil::ParseAsFloat("far", camera_json);
        SIM_INFO("camera init pos {} init focus {}", mCameraInitPos.transpose(),
                 mCameraInitFocus.transpose());
    }

    mCamera = std::make_shared<ArcBallCamera>(mCameraInitPos, mCameraInitFocus,
                                              tVector3f(0, 1, 0));
    mSimScene = cSceneBuilder::BuildSimScene(conf_path);

    mSimScene->Init(conf_path);

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
    mSimScene->Update(dt);
    DrawFrame();
}

void cDrawScene::Resize(int w, int h) { mFrameBufferResized = true; }

void cDrawScene::CursorMove(int xpos, int ypos)
{
    if (mLeftButtonPress)
    {
        mCamera->MouseMove(xpos, ypos);
    }
    mSimScene->CursorMove(this, xpos, ypos);
    // std::cout << "camera mouse move to " << xpos << " " << ypos << std::endl;
}

void cDrawScene::MouseButton(int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_1)
    {
        if (action == GLFW_RELEASE)
        {
            mLeftButtonPress = false;
            mCamera->ResetFlag();
        }

        else if (action == GLFW_PRESS)
        {
            mLeftButtonPress = true;
        }
    }
    mSimScene->MouseButton(this, button, action, mods);
}

void cDrawScene::Scroll(double xoff, double yoff)
{
    if (yoff > 0)
        mCamera->MoveForward();
    else if (yoff < 0)
        mCamera->MoveBackward();
}

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
    CreateDescriptorSetLayout();
    CreateGraphicsPipeline("triangle", mTriangleGraphicsPipeline);
    CreateGraphicsPipeline("line", mLinesGraphicsPipeline);
    CreateCommandPool();
    CreateDepthResources();
    CreateFrameBuffers();
    CreateTextureImage();
    CreateTextureImageView();
    CreateTextureSampler();
    CreateVertexBufferCloth();
    CreateVertexBufferGround();
    CreateLineBuffer();
    CreateMVPUniformBuffer();
    CreateDescriptorPool();
    CreateDescriptorSets();
    CreateCommandBuffers();
    CreateSemaphores();
}

extern uint32_t findMemoryType(VkPhysicalDevice phy_device, uint32_t typeFilter,
                               VkMemoryPropertyFlags props);
void cDrawScene::CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                              VkMemoryPropertyFlags props, VkBuffer &buffer,
                              VkDeviceMemory &buffer_memory)
{
    // 1. create a vertex buffer
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
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

VkCommandBuffer beginSingleTimeCommands(VkDevice device,
                                        VkCommandPool commandPool)
{
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    return commandBuffer;
}

void endSingleTimeCommands(VkDevice device, VkCommandPool commandPool,
                           VkQueue graphicsQueue, VkCommandBuffer commandBuffer)
{
    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);

    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

void cDrawScene::CopyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer,
                            VkDeviceSize size)
{
    auto cmd_buffer = beginSingleTimeCommands(mDevice, mCommandPool);
    // // 1. create a command buffer
    // VkCommandBufferAllocateInfo allo_info{};
    // allo_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    // allo_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    // allo_info.commandPool = mCommandPool;
    // allo_info.commandBufferCount = 1;
    // VkCommandBuffer cmd_buffer;
    // vkAllocateCommandBuffers(mDevice, &allo_info, &cmd_buffer);

    // // 2. begin to record the command buffer
    // VkCommandBufferBeginInfo begin_info{};
    // begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    // begin_info.flags =
    //     VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; // we only use this command buffer for a single time
    // vkBeginCommandBuffer(cmd_buffer, &begin_info);

    // 3. copy from src to dst buffer
    VkBufferCopy copy_region{};
    copy_region.srcOffset = 0;
    copy_region.dstOffset = 0;
    copy_region.size = size;
    vkCmdCopyBuffer(cmd_buffer, srcBuffer, dstBuffer, 1, &copy_region);

    endSingleTimeCommands(mDevice, mCommandPool, mGraphicsQueue, cmd_buffer);
    // // 4. end recording
    // vkEndCommandBuffer(cmd_buffer);

    // // 5. submit info
    // VkSubmitInfo submit_info{};
    // submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    // submit_info.commandBufferCount = 1;
    // submit_info.pCommandBuffers = &cmd_buffer;

    // vkQueueSubmit(mGraphicsQueue, 1, &submit_info, VK_NULL_HANDLE);

    // // wait, till the queue is empty (which means all commands have been finished)
    // vkQueueWaitIdle(mGraphicsQueue);

    // // 6. deconstruct
    // vkFreeCommandBuffers(mDevice, mCommandPool, 1, &cmd_buffer);
}

void cDrawScene::CreateVertexBufferCloth()
{
    const tVectorXf &draw_buffer = mSimScene->GetTriangleDrawBuffer();
    VkDeviceSize buffer_size = sizeof(float) * draw_buffer.size();
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
    memcpy(data, draw_buffer.data(), buffer_size);

    // unmap
    vkUnmapMemory(mDevice, stagingBufferMemory);

    CreateBuffer(buffer_size,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, mVertexBufferCloth,
                 mVertexBufferMemoryCloth);

    CopyBuffer(stagingBuffer, mVertexBufferCloth, buffer_size);
    vkDestroyBuffer(mDevice, stagingBuffer, nullptr);
    vkFreeMemory(mDevice, stagingBufferMemory, nullptr);
}

/**
 * \brief           draw a single frame
        1. get an image from the swap chain
        2. executate the command buffer with that image as attachment in the framebuffer
        3. return the image to the swap chain for presentation
    These 3 actions, ideally shvould be executed asynchronously. but the function calls will return before the operations are actually finished.
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

    // updating the uniform buffer values
    UpdateMVPUniformValue(imageIndex);
    UpdateVertexBufferCloth(imageIndex);
    UpdateVertexBufferGround(imageIndex);
    UpdateLineBuffer(imageIndex);

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
    for (const auto &x : requiredExtensions)
    {
        std::cout << "physical device lack extension " << x << std::endl;
    }

    // if required extensions are empty, means that all requred extensions are supported, return true;
    return requiredExtensions.empty();
}

void cDrawScene::CleanSwapChain()
{
    for (auto framebuffer : mSwapChainFramebuffers)
    {
        vkDestroyFramebuffer(mDevice, framebuffer, nullptr);
    }
    vkFreeCommandBuffers(mDevice, mCommandPool,
                         static_cast<uint32_t>(mCommandBuffers.size()),
                         mCommandBuffers.data());

    vkDestroyPipeline(mDevice, mTriangleGraphicsPipeline, nullptr);
    vkDestroyPipeline(mDevice, mLinesGraphicsPipeline, nullptr);
    vkDestroyPipelineLayout(mDevice, mPipelineLayout, nullptr);
    vkDestroyRenderPass(mDevice, mRenderPass, nullptr);

    for (auto imageView : mSwapChainImageViews)
    {
        vkDestroyImageView(mDevice, imageView, nullptr);
    }

    vkDestroySwapchainKHR(mDevice, mSwapChain, nullptr);

    for (size_t i = 0; i < mSwapChainImages.size(); i++)
    {
        vkDestroyBuffer(mDevice, mMVPUniformBuffers[i], nullptr);
        vkFreeMemory(mDevice, mMVPUniformBuffersMemory[i], nullptr);
    }

    vkDestroyDescriptorPool(mDevice, mDescriptorPool, nullptr);
}

struct MVPUniformBufferObject
{
    glm::vec4 camera_pos;
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
};

/**
 * \brief           Create Descriptor set layout
 *      The descriptor is used to store the uniform object used in the shader, we needs to spepcifiy how much uniform objects(descriptors), which is extacly "layout"
*/
void cDrawScene::CreateDescriptorSetLayout()
{
    // given the binding: offer the same info in C++ side as the shaders
    VkDescriptorSetLayoutBinding mvpLayoutBinding{};
    mvpLayoutBinding.binding = 0; // the binding info should be the same
    mvpLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    mvpLayoutBinding.descriptorCount = 1;
    mvpLayoutBinding.stageFlags =
        VK_SHADER_STAGE_VERTEX_BIT; // we use the descriptor in vertex shader
    mvpLayoutBinding.pImmutableSamplers = nullptr; // Optional

    // sampler
    VkDescriptorSetLayoutBinding samplerLayoutBinding{};
    samplerLayoutBinding.binding = 1;
    samplerLayoutBinding.descriptorCount = 1;
    samplerLayoutBinding.descriptorType =
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerLayoutBinding.pImmutableSamplers = nullptr;
    samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // std::array<VkDescriptorSetLayoutBinding, 2> bindings = {uboLayoutBinding, };
    // VkDescriptorSetLayoutCreateInfo layoutInfo{};
    // layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    // layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    // layoutInfo.pBindings = bindings.data();

    // create the layout
    std::array<VkDescriptorSetLayoutBinding, 2> ubo_set = {
        mvpLayoutBinding, samplerLayoutBinding};
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = ubo_set.size();
    layoutInfo.pBindings = ubo_set.data();

    if (vkCreateDescriptorSetLayout(mDevice, &layoutInfo, nullptr,
                                    &mDescriptorSetLayout) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create descriptor set layout!");
    }
    /// layout creation done
}

/**
 * \brief       Create buffer for uniform objects
*/
void cDrawScene::CreateMVPUniformBuffer()
{
    // check the size of uniform buffer object
    VkDeviceSize bufferSize = sizeof(MVPUniformBufferObject);

    // set up the memory
    mMVPUniformBuffers.resize(mSwapChainImages.size());
    mMVPUniformBuffersMemory.resize(mSwapChainImages.size());

    // create each uniform object buffer
    for (size_t i = 0; i < mSwapChainImages.size(); i++)
    {
        CreateBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     mMVPUniformBuffers[i], mMVPUniformBuffersMemory[i]);
    }
}

/**
 * \brief           Update the uniform value
 * 
 *  calculate the new uniform value and set them to the uniform buffer
*/
#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

glm::mat4 E2GLM(const tMatrix4f &em)
{
    glm::mat4 mat;
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            mat[j][i] = em(i, j);
        }
    }
    return mat;
}
void cDrawScene::UpdateMVPUniformValue(int image_idx)
{
    MVPUniformBufferObject ubo{};
    ubo.camera_pos =
        glm::vec4(mCamera->pos[0], mCamera->pos[1], mCamera->pos[2], 1.0f);
    // std::cout << "camera pos = " << mCamera->pos.transpose() << std::endl;
    ubo.model = glm::mat4(1.0f);
    tMatrix4f eigen_view = mCamera->ViewMatrix();
    ubo.view = E2GLM(eigen_view);
    ubo.proj = glm::perspective(
        glm::radians(fov),
        mSwapChainExtent.width / (float)mSwapChainExtent.height, near, far);
    ubo.proj[1][1] *= -1;

    void *data;
    vkMapMemory(mDevice, mMVPUniformBuffersMemory[image_idx], 0, sizeof(ubo), 0,
                &data);
    memcpy(data, &ubo, sizeof(ubo));
    vkUnmapMemory(mDevice, mMVPUniformBuffersMemory[image_idx]);
}

void cDrawScene::UpdateVertexBufferCloth(int image_idx)
{
    const tVectorXf &draw_buffer = mSimScene->GetTriangleDrawBuffer();
    // update
    VkDeviceSize buffer_size = sizeof(float) * draw_buffer.size();
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
    memcpy(data, draw_buffer.data(), buffer_size);

    // unmap
    vkUnmapMemory(mDevice, stagingBufferMemory);

    // CreateBuffer(buffer_size,
    //              VK_BUFFER_USAGE_TRANSFER_DST_BIT |
    //                  VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
    //              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, mVertexBufferCloth,
    //              mVertexBufferMemoryCloth);

    CopyBuffer(stagingBuffer, mVertexBufferCloth, buffer_size);
    vkDestroyBuffer(mDevice, stagingBuffer, nullptr);
    vkFreeMemory(mDevice, stagingBufferMemory, nullptr);
}

/**
 * \brief           connect the vkBuffer with the descriptor pool
*/
void cDrawScene::CreateDescriptorPool()
{
    // VkDescriptorPoolSize poolSize{};
    // poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    // poolSize.descriptorCount = static_cast<uint32_t>(mSwapChainImages.size());

    // VkDescriptorPoolCreateInfo poolInfo{};
    // poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    // poolInfo.poolSizeCount = 1;
    // poolInfo.pPoolSizes = &poolSize;

    // poolInfo.maxSets = static_cast<uint32_t>(mSwapChainImages.size());

    // if (vkCreateDescriptorPool(mDevice, &poolInfo, nullptr, &mDescriptorPool) != VK_SUCCESS)
    // {
    //     throw std::runtime_error("failed to create descriptor pool!");
    // }
    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount =
        static_cast<uint32_t>(mSwapChainImages.size());
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount =
        static_cast<uint32_t>(mSwapChainImages.size());

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = static_cast<uint32_t>(mSwapChainImages.size());

    if (vkCreateDescriptorPool(mDevice, &poolInfo, nullptr, &mDescriptorPool) !=
        VK_SUCCESS)
    {
        throw std::runtime_error("failed to create descriptor pool!");
    }
}

void cDrawScene::CreateDescriptorSets()

{
    // std::cout << "begin to create descriptor set\n";
    // create the descriptor set
    std::vector<VkDescriptorSetLayout> layouts(mSwapChainImages.size(),
                                               mDescriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = mDescriptorPool;
    allocInfo.descriptorSetCount =
        static_cast<uint32_t>(mSwapChainImages.size());
    allocInfo.pSetLayouts = layouts.data();

    mDescriptorSets.resize(mSwapChainImages.size());
    if (vkAllocateDescriptorSets(mDevice, &allocInfo, mDescriptorSets.data()) !=
        VK_SUCCESS)
    {
        throw std::runtime_error("failed to allocate descriptor sets!");
    }
    // create each descriptor
    // mSwapChainImages; mUniformBuffers;
    for (size_t i = 0; i < mSwapChainImages.size(); i++)
    {
        // printf("---------------%d----------------\n", i);
        VkDescriptorBufferInfo mvpbufferinfo{};
        mvpbufferinfo.buffer = mMVPUniformBuffers[i];
        mvpbufferinfo.offset = 0;
        mvpbufferinfo.range = sizeof(MVPUniformBufferObject);

        VkDescriptorImageInfo imageInfo{};
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo.imageView = mTextureImageView;
        imageInfo.sampler = mTextureSampler;

        std::array<VkWriteDescriptorSet, 2> descriptorWrites{};

        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = mDescriptorSets[i];
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &mvpbufferinfo;

        {

            descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[1].dstSet = mDescriptorSets[i];
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorType =
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[1].descriptorCount = 1;
            descriptorWrites[1].pImageInfo = &imageInfo;
        }
        vkUpdateDescriptorSets(mDevice, descriptorWrites.size(),
                               descriptorWrites.data(), 0, nullptr);
    }
    // std::cout << "end to create descriptor set\n";
    // exit(0);
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

        std::array<VkClearValue, 2> clearValues{};
        clearValues[0].color = {1.0f, 1.0f, 1.0f, 1.0f};
        clearValues[1].depthStencil = {1.0f, 0};

        renderPassInfo.clearValueCount =
            static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

        // full black clear color
        // VkClearValue clear_color = {1.0f, 1.0f, 1.0f, 1.0f};
        renderPassInfo.clearValueCount = clearValues.size();
        renderPassInfo.pClearValues = clearValues.data();

        // begin render pass
        vkCmdBeginRenderPass(mCommandBuffers[i], &renderPassInfo,
                             VK_SUBPASS_CONTENTS_INLINE);

        CreateTriangleCommandBuffers(i);
        CreateLineCommandBuffers(i);

        // end render pass
        vkCmdEndRenderPass(mCommandBuffers[i]);

        SIM_ASSERT(VK_SUCCESS == vkEndCommandBuffer(mCommandBuffers[i]));
    }
    SIM_INFO("Create Command buffers succ");
}

void cDrawScene::CreateVertexBufferGround()
{
    VkDeviceSize buffer_size =
        sizeof(ground_vertices[0]) * ground_vertices.size();

    // 5. copy the vertex data to the buffer
    CreateBuffer(buffer_size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 mVertexBufferGround, mVertexBufferMemoryGround);
}

void cDrawScene::UpdateVertexBufferGround(int idx)
{
    // update
    VkDeviceSize buffer_size =
        sizeof(ground_vertices[0]) * ground_vertices.size();

    // 5. copy the vertex data to the buffer
    void *data = nullptr;
    // map the memory to "data" ptr;
    vkMapMemory(mDevice, mVertexBufferMemoryGround, 0, buffer_size, 0, &data);

    // write the data
    memcpy(data, ground_vertices.data(), buffer_size);

    // unmap
    vkUnmapMemory(mDevice, mVertexBufferMemoryGround);
}

void cDrawScene::CreateTriangleCommandBuffers(int i)
{
    vkCmdBindPipeline(mCommandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
                      mTriangleGraphicsPipeline);
    {
        VkBuffer vertexBuffers[] = {mVertexBufferGround};
        // VkBuffer vertexBuffers[] = {mVertexBufferGround, mVertexBufferCloth};
        VkDeviceSize offsets[] = {0};
        vkCmdBindVertexBuffers(mCommandBuffers[i], 0, 1, vertexBuffers,
                               offsets);

        // update the uniform objects (descriptors)
        vkCmdBindDescriptorSets(
            mCommandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
            mPipelineLayout, 0, 1, &mDescriptorSets[i], 0, nullptr);

        // draaaaaaaaaaaaaaaaaaaaaaaaaaaaw!
        // uint32_t triangle_size =  / 3;

        vkCmdDraw(mCommandBuffers[i], ground_vertices.size(), 1, 0, 0);
    }
    {
        VkBuffer vertexBuffers[] = {mVertexBufferCloth};
        // VkBuffer vertexBuffers[] = {, mVertexBufferCloth};
        VkDeviceSize offsets[] = {0};
        vkCmdBindVertexBuffers(mCommandBuffers[i], 0, 1, vertexBuffers,
                               offsets);

        // update the uniform objects (descriptors)
        vkCmdBindDescriptorSets(
            mCommandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
            mPipelineLayout, 0, 1, &mDescriptorSets[i], 0, nullptr);

        // draaaaaaaaaaaaaaaaaaaaaaaaaaaaw!
        // uint32_t triangle_size =  / 3;
        const tVectorXf &draw_buffer = mSimScene->GetTriangleDrawBuffer();
        SIM_ASSERT(draw_buffer.size() % 3 == 0);
        vkCmdDraw(mCommandBuffers[i], draw_buffer.size() / 3, 1, 0, 0);
    }
}
