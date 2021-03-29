
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
#include "scenes/SceneBuilder.h"
GLFWwindow *window = nullptr;
std::shared_ptr<cDrawScene> scene = nullptr;
bool esc_pushed = false;

static void ResizeCallback(GLFWwindow *window, int w, int h)
{
    scene->Resize(w, h);
}

static void CursorPositionCallback(GLFWwindow *window, double xpos, double ypos)
{
    scene->CursorMove(xpos, ypos);
}

void mouse_button_callback(GLFWwindow *window, int button, int action, int mods)
{
    scene->MouseButton(button, action, mods);
}
void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    {
        esc_pushed = true;
    }
}

void InitGlfw()
{
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window = glfwCreateWindow(800, 600, "Vulkan window", nullptr, nullptr);
    glfwSetFramebufferSizeCallback(window, ResizeCallback);
    glfwSetCursorPosCallback(window, CursorPositionCallback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetKeyCallback(window, key_callback);
}

#include "utils/LogUtil.h"
int main()
{
    InitGlfw();
    scene = cSceneBuilder::BuildScene("cloth_sim_draw");
    scene->Init("config/conf.json");

    double dt = 1e-3;
    while (glfwWindowShouldClose(window) == false && esc_pushed == false)
    {
        glfwPollEvents();
        scene->Update(dt);
    }
    glfwDestroyWindow(window);

    glfwTerminate();
    return 0;
}