#pragma once


#include <iostream>
#include <condition_variable>
#include <memory>
#include <thread>
#include <string>
#include <future>
#include <mutex>
#include <atomic>
#include <iostream>
#include <sstream>
#include <vector>
#include <array>
#include <optional>
#include <variant>
#include <stdexcept>
#include <cstdint>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <random>

#include <boost/noncopyable.hpp>
#define CL_TARGET_OPENCL_VERSION 200
#define BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION
#include <boost/compute.hpp>