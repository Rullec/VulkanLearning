
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
// #define GLFW_EXPOSE_NATIVE_
#include <GLFW/glfw3native.h>
#endif

#include <iostream>
#include <memory>
#include "scenes/DrawScene.h"
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