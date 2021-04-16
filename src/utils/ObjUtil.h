#pragma once
#include <string>

class cObjUtil
{
public:
    struct tParams
    {
        std::string mPath; // obj file path
    };

    static void LoadObj(const tParams &param);
};