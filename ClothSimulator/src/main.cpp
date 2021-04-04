
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

#include "scenes/SceneBuilder.h"
#include <iostream>
#include <memory>
GLFWwindow *window = nullptr;
std::shared_ptr<cDrawScene> scene = nullptr;
bool esc_pushed = false;
bool gPause = false;
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
void key_callback(GLFWwindow *window, int key, int scancode, int action,
                  int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    {
        esc_pushed = true;
    }
    else if (key == GLFW_KEY_I && action == GLFW_PRESS)
    {
        gPause = !gPause;
        std::cout << "[log] simulation paused\n";
    }
}

void scroll_callback(GLFWwindow *window, double xoffset, double yoffset)
{
    scene->Scroll(xoffset, yoffset);
}
void InitGlfw()
{
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window = glfwCreateWindow(800, 600, "Cloth Simulator", nullptr, nullptr);
    glfwSetFramebufferSizeCallback(window, ResizeCallback);
    glfwSetCursorPosCallback(window, CursorPositionCallback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetKeyCallback(window, key_callback);
    glfwSetScrollCallback(window, scroll_callback);
}

#include "utils/LogUtil.h"
#include "utils/TimeUtil.hpp"
int main()
{
    InitGlfw();
    scene = cSceneBuilder::BuildScene("cloth_sim_draw");
    scene->Init("config/config.json");

    auto last = cTimeUtil::GetCurrentTime();
    while (glfwWindowShouldClose(window) == false && esc_pushed == false)
    {
        glfwPollEvents();

        // 1. calc delta time for real time simulation
        auto cur = cTimeUtil::GetCurrentTime();
        double delta_time = cTimeUtil::CalcTimeElaspedms(last, cur) * 1e-3;

        // 2. update
        // delta_time = 1e-3;
        // delta_time /= 4;
        double limit = 1.0 / 10;
        // double limit = 1e-4;
        delta_time = std::min(delta_time, limit);
        if (gPause == false)
        {
            scene->Update(delta_time);
        }

        last = cur;
    }

    glfwDestroyWindow(window);

    glfwTerminate();
    return 0;
}