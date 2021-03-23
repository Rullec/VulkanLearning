#pragma once
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
	VkInstance mInstance;
};
