//#pragma once
#include <vulkan/vulkan.h>
#include <vector>

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
	void Reset();

protected:
	void InitVulkan();

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
	VkShaderModule CreateShaderModule(const std::vector<char>& code);
	void CreateRenderPass();
	VkInstance mInstance;
	VkPhysicalDevice mPhysicalDevice;
	VkDevice mDevice;		// logical device
	VkSurfaceKHR mSurface;	// window surface
	VkQueue mGraphicsQueue; // device queue (only one)
	VkQueue mPresentQueue;
	VkSwapchainKHR mSwapChain; // swapchain, literally frame buffer
	std::vector<VkImage> mSwapChainImages;
	std::vector<VkImageView> mSwapChainImageViews;
	VkFormat mSwapChainImageFormat;
	VkExtent2D mSwapChainExtent;
	VkPipelineLayout mPipelineLayout;	// uniform values in the shader
	VkRenderPass mRenderPass;	// special settings for a render pass
	//VkPipelineLayout mPipelineLayout;	// 
	VkPipeline mGraphicsPipeline;
};