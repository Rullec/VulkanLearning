//#pragma once
#include <vector>
#include <vulkan/vulkan.h>
#include "Scene.h"
/**
 * \brief			Main Vulkan Draw Scene for cloth simulator
*/
class cSimScene;
class ArcBallCamera;
class cDrawScene : public cScene
{
public:
    explicit cDrawScene();
    virtual ~cDrawScene();
    virtual void Init(const std::string &conf_path) override final;
    virtual void Update(double dt) override final;
    void MainLoop();
    void Resize(int w, int h);
    void CursorMove(int xpos, int ypos);
    void MouseButton(int button, int action, int mods);
    virtual void Reset() override final;

protected:
    void InitVulkan();

    void DrawFrame();
    void CleanVulkan();

private:
    void CreateInstance();
    void CheckAvaliableExtensions() const;
    bool CheckValidationLayerSupport() const;
    void SetupDebugMessenger();
    void PickPhysicalDevice();
    void CreateLogicalDevice();
    void CreateSurface();
    void CreateSwapChain();
    void CreateImageViews(); // create "view" for swap chain images
    void CreateGraphicsPipeline();
    VkShaderModule CreateShaderModule(const std::vector<char> &code);
    void CreateRenderPass();
    void CreateFrameBuffers();
    void CreateCommandPool();
    void CreateCommandBuffers();
    void CreateSemaphores();
    void RecreateSwapChain();
    void CleanSwapChain();
    void CreateVertexBufferCloth();
    void CreateVertexBufferGround();
    void CreateUniformBuffer();
    void CopyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
    void CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags props, VkBuffer &buffer,
                      VkDeviceMemory &buffer_memory);
    void CreateDescriptorSetLayout();
    void UpdateUniformValue(int image_idx);
    void UpdateVertexBufferCloth(int idx);
    void UpdateVertexBufferGround(int idx);

    void CreateDescriptorPool();
    void CreateDescriptorSets();
    VkInstance mInstance;
    VkPhysicalDevice mPhysicalDevice;
    VkDevice mDevice;       // logical device
    VkSurfaceKHR mSurface;  // window surface
    VkQueue mGraphicsQueue; // device queue (only one)
    VkQueue mPresentQueue;
    VkSwapchainKHR mSwapChain; // swapchain, literally frame buffer
    std::vector<VkImage> mSwapChainImages;
    std::vector<VkImageView> mSwapChainImageViews;
    VkFormat mSwapChainImageFormat;
    VkExtent2D mSwapChainExtent;
    VkDescriptorSetLayout mDescriptorSetLayout; // descriptors (uniform objects) layout used in the shader
    VkPipelineLayout mPipelineLayout;           // uniform values in the shader
    VkRenderPass mRenderPass;                   // special settings for a render pass
    VkPipeline mGraphicsPipeline;
    std::vector<VkFramebuffer> mSwapChainFramebuffers; //
    VkCommandPool mCommandPool;
    std::vector<VkCommandBuffer>
        mCommandBuffers; // each command buffer(queue) serves a frame buffer

    std::vector<VkSemaphore>
        mImageAvailableSemaphore; // the image acquired from the swap chain is ready to be rendered to
    std::vector<VkSemaphore>
        mRenderFinishedSemaphore; // the rendering done, the image can be sent back to the swap chain for presentation on the screen.
    std::vector<VkFence>
        minFlightFences; // fences to do CPU-GPU synchronization
    std::vector<VkFence> mImagesInFlight;
    int mCurFrame;
    bool mFrameBufferResized;
    VkBuffer mVertexBufferCloth;
    VkDeviceMemory mVertexBufferMemoryCloth;

    VkBuffer mVertexBufferGround;
    VkDeviceMemory mVertexBufferMemoryGround;
    // buffers used for uniform objects
    std::vector<VkBuffer> mUniformBuffers;             // MVP uniform buffer
    std::vector<VkDeviceMemory> mUniformBuffersMemory; // their memories
    VkDescriptorPool mDescriptorPool;
    std::vector<VkDescriptorSet> mDescriptorSets; // real descriptor
    std::shared_ptr<ArcBallCamera> mCamera;

    // simulation scene
    std::shared_ptr<cSimScene> mSimScene;

    bool mButtonPress;
};