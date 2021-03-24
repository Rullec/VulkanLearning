//#pragma once
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
	VkInstance mInstance;
	VkPhysicalDevice mPhysicalDevice;
	VkDevice mDevice;		// logical device
	VkSurfaceKHR mSurface;	// window surface
	VkQueue mGraphicsQueue; // device queue (only one)
	VkQueue mPresentQueue;
	VkSwapchainKHR mSwapChain;
};