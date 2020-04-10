#include "pch.hpp"
#include "catch.hpp"

inline const std::string BITONIC_SORT_KERNEL_STRING =
#include "../GPUSorter/bitonic_sort.cl"
"";

template<typename T, typename SeedT = std::default_random_engine::result_type>
auto generate_uniformly_distributed_vec(std::size_t count, T min, T max, SeedT seed = std::default_random_engine::default_seed)
{
    std::default_random_engine generator{ seed };
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    std::vector<T> vals(count);

    double scale = max - min;


    assert(scale > T{});
    assert(scale > T{});

    for (int i = 0; i < count; ++i)
    {
        vals[i] = min + static_cast<T>(scale * distribution(generator));
    }

    return vals;
}

namespace bc = boost::compute;

TEST_CASE("bitonic sort", "[algo] [sort]")
{
    auto queue = boost::compute::system::default_queue();
    auto host_vec = generate_uniformly_distributed_vec(512, -135416.0f, 41230.0f);

    //host_vec[0] = host_vec[231];
    //host_vec[5] = host_vec[234];

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

    //std::vector<float> input(result.size());
    //bc::copy(host_vec.begin(), host_vec.begin() + input.size(), input.begin());

    auto check = std::is_sorted(result.begin(), result.end());
    REQUIRE(check);
}
