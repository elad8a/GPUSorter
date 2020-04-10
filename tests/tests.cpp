#include "pch.hpp"
#pragma comment(lib, "opencl.lib")
#ifndef CATCH_CONFIG_RUNNER
#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

int main(int argc, char* argv[])
{
    Catch::Session().run(argc, argv);
}

#endif