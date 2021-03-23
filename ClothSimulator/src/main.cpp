#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include <scenes/DrawScene.h>
#include <iostream>
GLFWwindow *window;
void InitGlfw()
{
	glfwInit();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	window = glfwCreateWindow(800, 600, "Vulkan window", nullptr, nullptr);
}
int main()
{
	InitGlfw();
	std::shared_ptr<cDrawScene> scene = std::make_shared<cDrawScene>();
	scene->Init();
	double cur_time = glfwGetTime();
	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();
		double now_time = glfwGetTime();
		scene->Update(now_time - cur_time);
		//std::cout << "cost " << now_time - cur_time << std::endl;
		cur_time = now_time;
	}

	glfwDestroyWindow(window);

	glfwTerminate();

	return 0;
}