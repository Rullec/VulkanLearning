#include <chrono>
#include <ctime>
#include <map>
#include <string>
class cTimeUtil
{
public:
    // calculate a continuous segment of time
    static void Begin(const std::string &name);
    static void End(const std::string &name);

    // calculate a discrete segment of time, lazy calculation until the final
    static void BeginLazy(const std::string &name);
    static void EndLazy(const std::string &name);
    static void ClearLazy(const std::string &name);

    static std::string GetSystemTime();

private:
    inline static std::map<const std::string,
                           std::chrono::high_resolution_clock::time_point>
        mTimeTable; // record current time
    inline static std::map<
        const std::string,
        std::chrono::high_resolution_clock::time_point>::iterator time_it;

    inline static std::map<const std::string, double>
        mLazyTimeTable; // record lazy accumulated time
};