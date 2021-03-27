//#pragma once
#include <vector>
#include <vulkan/vulkan.h>

/**
 * \brief			Main Vulkan Draw Scene for cloth simulator
*/
class cDrawScene
{
public:
    explicit cDrawScene();
    virtual ~cDrawScene();
    void Init();
    void Update(double dt);
    void MainLoop();
    void Resize(int w, int h);
    void Reset();

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
    void CreateVertexBuffer();
    void CopyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
    void CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                              VkMemoryPropertyFlags props, VkBuffer &buffer,
                              VkDeviceMemory &buffer_memory);
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
    VkPipelineLayout mPipelineLayout; // uniform values in the shader
    VkRenderPass mRenderPass;         // special settings for a render pass
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
    VkBuffer mVertexBuffer;
    VkDeviceMemory mVertexBufferMemory;
};