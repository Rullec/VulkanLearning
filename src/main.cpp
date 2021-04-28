
#ifdef _WIN32
#define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_INCLUDE_VULKAN
#include "GLFW/glfw3.h"
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
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

#include "utils/FileUtil.h"
#include "utils/JsonUtil.h"
#include "utils/LogUtil.h"
#include <cmath>

GLFWwindow *window = nullptr;
std::shared_ptr<cDrawScene> draw_scene = nullptr;
std::shared_ptr<cScene> scene = nullptr;
bool esc_pushed = false;
bool gPause = true;
int gWindowWidth, gWindowHeight;

static void ResizeCallback(GLFWwindow *window, int w, int h)
{
    draw_scene->Resize(w, h);
}

static void CursorPositionCallback(GLFWwindow *window, double xpos, double ypos)
{
    draw_scene->CursorMove(xpos, ypos);
}

void MouseButtonCallback(GLFWwindow *window, int button, int action, int mods)
{
    draw_scene->MouseButton(button, action, mods);
}
void KeyCallback(GLFWwindow *window, int key, int scancode, int action,
                 int mods)
{
    draw_scene->Key(key, scancode, action, mods);
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

void ScrollCallback(GLFWwindow *window, double xoffset, double yoffset)
{
    draw_scene->Scroll(xoffset, yoffset);
}
void InitGlfw()
{
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window = glfwCreateWindow(gWindowWidth, gWindowHeight, "Cloth Simulator", nullptr, nullptr);
    glfwSetFramebufferSizeCallback(window, ResizeCallback);
    glfwSetCursorPosCallback(window, CursorPositionCallback);
    glfwSetMouseButtonCallback(window, MouseButtonCallback);
    glfwSetKeyCallback(window, KeyCallback);
    glfwSetScrollCallback(window, ScrollCallback);
}

#include "utils/LogUtil.h"
#include "utils/TimeUtil.hpp"
bool gEnableDraw = true;
// bool gEnableDraw = false;
void SimDraw(const std::string &conf_path);
void SimNoDraw(const std::string &conf_path);
void ParseConfig(std::string conf);
int main(int argc, char **argv)
{
    SIM_ASSERT(argc == 2);
    std::string conf = std::string(argv[1]);
    ParseConfig(conf);

    if (gEnableDraw == true)
    {
        SimDraw(conf);
    }
    else
    {
        SimNoDraw(conf);
    }
    return 0;
}

void SimDraw(const std::string &conf)
{
    InitGlfw();
    scene = cSceneBuilder::BuildScene("cloth_sim_draw");
    draw_scene = std::dynamic_pointer_cast<cDrawScene>(scene);
    draw_scene->Init(conf);
    auto last = cTimeUtil::GetCurrentTime_chrono();
    while (glfwWindowShouldClose(window) == false && esc_pushed == false)
    {
        glfwPollEvents();

        // 1. calc delta time for real time simulation
        auto cur = cTimeUtil::GetCurrentTime_chrono();
        double delta_time = cTimeUtil::CalcTimeElaspedms(last, cur) * 1e-3;

        // 2. update
        // delta_time = 1e-3;
        // delta_time /= 4;
        double limit = 1.0 / 30;
        // double limit = 1e-4;
#ifdef _WIN32
        delta_time = std::min(delta_time, limit);
#else
        delta_time = std::min(delta_time, limit);
#endif

        // delta_time = 1e-4;
        if (gPause == false)
        {
            // cTimeUtil::Begin("scene_update");
            draw_scene->Update(delta_time);
            // cTimeUtil::End("scene_update");
        }

        last = cur;
    }

    glfwDestroyWindow(window);

    glfwTerminate();
}

void SimNoDraw(const std::string &conf_path)
{
    // InitGlfw();
    scene = cSceneBuilder::BuildSimScene(conf_path);
    scene->Init(conf_path);

    int max_iters = 1e3;
    int cur_iter = 0;
    double dt = 1e-2;
    while (++cur_iter < max_iters)
    {
        scene->Update(dt);
        printf("[debug] iters %d/%d\n", cur_iter, max_iters);
    }
}
void ParseConfig(std::string conf)
{
    SIM_ASSERT(cFileUtil::ExistsFile(conf) == true);
    Json::Value root;
    cJsonUtil::LoadJson(conf, root);
    gPause = cJsonUtil::ParseAsBool("pause_at_first", root);
    gEnableDraw = cJsonUtil::ParseAsBool("enable_draw", root);
    if (gEnableDraw == true)
    {
        gWindowWidth = cJsonUtil::ParseAsInt("window_width", root);
        gWindowHeight = cJsonUtil::ParseAsInt("window_height", root);
    }
    SIM_INFO("pause at first = {}", gPause);
}