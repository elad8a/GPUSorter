#include "pch.hpp"
#include "catch.hpp"
#include "tests.utils.hpp"
#include "sorting_kernels.hpp"

namespace bc = boost::compute;

TEST_CASE("bitonic sort - 512 elements", "[algo] [sort]")
{
    auto queue = boost::compute::system::default_queue();
    auto host_vec = generate_uniformly_distributed_vec(512, -135416.0f, 41230.0f);

    bc::vector<float> device_vec(host_vec.begin(), host_vec.end());
    auto program = bc::program::build_with_source(
        BITONIC_SORT_KERNEL_STRING, 
        boost::compute::system::default_context(),
        "-cl-std=CL2.0"
        );

    auto kernel = program.create_kernel("bitonic_sort");
    kernel.set_arg(0, device_vec);
    kernel.set_arg(1, static_cast<unsigned>(device_vec.size()));
    queue.enqueue_1d_range_kernel(
        kernel,
        0, 256, 256);

    std::vector<float> result(host_vec.size());
    bc::copy(device_vec.begin(), device_vec.begin() + result.size(), result.begin());

    auto check = std::is_sorted(result.begin(), result.end());
    REQUIRE(check);
}

TEST_CASE("bitonic sort - less then 512 elements", "[algo] [sort]")
{
    auto queue = boost::compute::system::default_queue();
    auto host_vec = generate_uniformly_distributed_vec(412, -135416.0f, 41230.0f);

    bc::vector<float> device_vec(host_vec.begin(), host_vec.end());
    auto program = bc::program::build_with_source(
        BITONIC_SORT_KERNEL_STRING,
        boost::compute::system::default_context(),
        "-cl-std=CL2.0"
        );

    auto kernel = program.create_kernel("bitonic_sort");
    kernel.set_arg(0, device_vec);
    kernel.set_arg(1, static_cast<unsigned>(device_vec.size()));
    queue.enqueue_1d_range_kernel(
        kernel,
        0, 256, 256);

    std::vector<float> result(host_vec.size());
    bc::copy(device_vec.begin(), device_vec.begin() + result.size(), result.begin());

    auto check = std::is_sorted(result.begin(), result.end());
    REQUIRE(check);
}

TEST_CASE("batched bitonic sort - 512 elements", "[algo] [sort]")
{
    auto single_batch_size = 512u;
    auto batch_count = 200u;
    auto queue = boost::compute::system::default_queue();
    auto host_vec = generate_uniformly_distributed_vec(single_batch_size * batch_count, -135416.0f, 41230.0f);

    bc::vector<float> device_vec(host_vec.begin(), host_vec.end());
    auto program = bc::program::build_with_source(
        BITONIC_SORT_KERNEL_STRING,
        boost::compute::system::default_context(),
        "-cl-std=CL2.0"
        );

    auto kernel = program.create_kernel("bitonic_sort_batched");
    kernel.set_arg(0, device_vec);
    kernel.set_arg(1, single_batch_size);
    queue.enqueue_1d_range_kernel(
        kernel,
        0, 256 * batch_count, 256);

    std::vector<float> result(host_vec.size());
    bc::copy(device_vec.begin(), device_vec.begin() + result.size(), result.begin());

    for (auto i = 0u; i < batch_count; ++i)
    {
        auto check = std::is_sorted(
            result.begin() + i * single_batch_size,
            result.begin() + (i + 1) * single_batch_size
            );

        REQUIRE(check);
    }

}

TEST_CASE("batched bitonic sort - less than 512 elements", "[algo] [sort]")
{
    auto single_batch_size = 123u;
    auto batch_count = 213u;
    auto queue = boost::compute::system::default_queue();
    auto host_vec = generate_uniformly_distributed_vec(single_batch_size * batch_count, -135416.0f, 41230.0f);

    bc::vector<float> device_vec(host_vec.begin(), host_vec.end());
    auto program = bc::program::build_with_source(
        BITONIC_SORT_KERNEL_STRING,
        boost::compute::system::default_context(),
        "-cl-std=CL2.0"
        );

    auto kernel = program.create_kernel("bitonic_sort_batched");
    kernel.set_arg(0, device_vec);
    kernel.set_arg(1, single_batch_size);
    queue.enqueue_1d_range_kernel(
        kernel,
        0, 256 * batch_count, 256);

    std::vector<float> result(host_vec.size());
    bc::copy(device_vec.begin(), device_vec.begin() + result.size(), result.begin());

    for (auto i = 0u; i < batch_count; ++i)
    {
        auto check = std::is_sorted(
            result.begin() + i * single_batch_size,
            result.begin() + (i + 1) * single_batch_size
            );

        REQUIRE(check);
    }

}


