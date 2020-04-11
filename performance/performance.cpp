#include "pch.hpp"
#pragma comment(lib, "opencl.lib")
#pragma comment(lib, "/nvtx/lib/x64/nvToolsExt64_1.lib")

#ifndef CATCH_CONFIG_RUNNER
#define CATCH_CONFIG_RUNNER
#include "catch.hpp"


int main(int argc, char* argv[])
{
    Catch::Session().run(argc, argv);
}

#endif