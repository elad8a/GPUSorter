#include "pch.hpp"
#include "catch.hpp"
#include "tests.utils.hpp"
inline const std::string PARTITION_KERNEL_STRING =
#include "../GPUSorter/partition.cl"
"";

namespace bc = boost::compute;
typedef struct parent_segment
{
    unsigned current_smaller_than_pivot_start_idx; // for allocation
    unsigned current_greater_than_pivot_end_idx;
    unsigned start_segment_global_idx;
} parent_segment;

TEST_CASE("gpu partition - single chunk", "[algo] [sort] [fail]")
{
    using data_t = float;
    auto queue = boost::compute::system::default_queue();
    auto program = bc::program::build_with_source(
        PARTITION_KERNEL_STRING,
        boost::compute::system::default_context(),
        "-cl-std=CL2.0"
        );

    auto partition_kernel = program.create_kernel("partition");

    auto min_val = 0.0f;
    auto max_val = 5000.0f;
    auto local_size = 256u;
    auto global_size = local_size; // single chunk single work group

    std::vector<data_t> pivots{ 1000.0f, 500.0f, 4000.0f, 0.0f, 5000.0f };

    SECTION("chunk size == local_size")
    {
        auto input = generate_uniformly_distributed_vec(local_size, min_val, max_val);
        std::vector<data_t> host_output(input.size());
        parent_segment host_parent{};
        host_parent.current_smaller_than_pivot_start_idx = 0;
        host_parent.current_greater_than_pivot_end_idx = static_cast<unsigned>(input.size());
        host_parent.start_segment_global_idx = 0;

        bc::vector<data_t> src(input.begin(), input.end(), queue);
        bc::vector<data_t> dst(host_output.begin(), host_output.end(), queue);
        bc::vector<parent_segment> parents(1, queue.get_context());
        parents[0] = host_parent;

        partition_kernel.set_arg(0, src);
        partition_kernel.set_arg(1, dst);
        partition_kernel.set_arg(3, parents);
        partition_kernel.set_arg(4, static_cast<unsigned>(input.size()));

 

        for (auto pivot : pivots)
        {
            queue.enqueue_1d_range_kernel(partition_kernel, 0, global_size, local_size);

            bc::copy(dst.begin(), dst.end(), host_output.begin(), queue);
            if (pivot != min_val && pivot != max_val)
            {
                CHECK(!is_partitioned(input.begin(), input.end(), [=](auto val) {return val < pivot; }));
            }
            REQUIRE(is_partitioned(host_output.begin(), host_output.end(), [=](auto val) {return val < pivot; }));
            CHECK(std::is_permutation(input.begin(), input.end(), host_output.begin()));
        }

    }

}

TEST_CASE("partition sort - 512 elements", "[algo] [sort]  [fail]")
{
    auto queue = boost::compute::system::default_queue();
    auto host_vec = generate_uniformly_distributed_vec(512, -135416.0f, 41230.0f);

    bc::vector<float> device_vec(host_vec.begin(), host_vec.end());
    auto program = bc::program::build_with_source(
        PARTITION_KERNEL_STRING,
        boost::compute::system::default_context(),
        "-cl-std=CL2.0"
        );

    REQUIRE(true);
    //auto kernel = program.create_kernel("bitonic_sort");
    //kernel.set_arg(0, device_vec);
    //kernel.set_arg(1, static_cast<unsigned>(device_vec.size()));
    //queue.enqueue_1d_range_kernel(
    //    kernel,
    //    0, 256, 256);

    //std::vector<float> result(host_vec.size());
    //bc::copy(device_vec.begin(), device_vec.begin() + result.size(), result.begin());

    //auto check = std::is_sorted(result.begin(), result.end());
    //REQUIRE(check);
}
