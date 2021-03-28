
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

#ifdef __APPLE__
#include <GLFW/glfw3.h>
#endif

#include <iostream>
#include <memory>
#include "scenes/DrawScene.h"
GLFWwindow *window;
std::shared_ptr<cDrawScene> scene = nullptr;

static void ResizeCallback(GLFWwindow *window, int w, int h)
{
    scene->Resize(w, h);
}

void InitGlfw()
{
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window = glfwCreateWindow(800, 600, "Vulkan window", nullptr, nullptr);
    glfwSetFramebufferSizeCallback(window, ResizeCallback);
}

int main()
{
    InitGlfw();
    scene = std::make_shared<cDrawScene>();
    scene->Init();

    double dt = 1e-3;
    while (glfwWindowShouldClose(window) == false)
    {
        glfwPollEvents();
        scene->Update(dt);
    }
    return 0;
}