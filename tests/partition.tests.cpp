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

TEST_CASE("gpu partition", "[algo] [sort]")
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

    std::vector<data_t> input1 = generate_uniformly_distributed_vec(local_size, min_val, max_val);
    std::vector<data_t> input2 = generate_uniformly_distributed_vec(local_size - 50, min_val, max_val);
    std::vector<data_t> input3 = generate_uniformly_distributed_vec(local_size * 4 - 13, min_val, max_val);

    

    auto checker = [&](std::vector<data_t>& input)
    {
        std::vector<data_t> host_output(input.size());
        parent_segment host_parent{};
        host_parent.current_smaller_than_pivot_start_idx = 0;
        host_parent.current_greater_than_pivot_end_idx = static_cast<unsigned>(input.size());
        host_parent.start_segment_global_idx = 0;

        bc::vector<data_t> src(input.begin(), input.end(), queue);
        bc::vector<data_t> dst(host_output.begin(), host_output.end(), queue);
        bc::vector<parent_segment> parents(1, queue.get_context());


        partition_kernel.set_arg(0, src);
        partition_kernel.set_arg(1, dst);
        partition_kernel.set_arg(2, parents);
        partition_kernel.set_arg(3, static_cast<unsigned>(input.size()));



        for (auto pivot : pivots)
        {
            parents[0] = host_parent;
            partition_kernel.set_arg(4, pivot);
            queue.enqueue_1d_range_kernel(partition_kernel, 4, global_size, local_size);

            bc::copy(dst.begin(), dst.end(), host_output.begin(), queue);
            if (pivot != min_val && pivot != max_val)
            {
                CHECK(!is_partitioned(input.begin(), input.end(), [=](auto val) {return val < pivot; }));
            }
            auto is_partition = is_partitioned(host_output.begin(), host_output.end(), [=](auto val) {return val < pivot; });
            REQUIRE(is_partition);
            //CHECK(std::is_permutation(input.begin(), input.end(), host_output.begin()));
        }
    };

    SECTION("segment size == local_size")
    {
        checker(input1);
    }
    SECTION("segment size < local_size")
    {
        checker(input2);
    }
    SECTION("segment size > local_size")
    {
        checker(input3);
    }


}


TEST_CASE("gpu partition - batched", "[algo] [sort] [fail]")
{
    using data_t = float;
    auto queue = boost::compute::system::default_queue();
    auto program = bc::program::build_with_source(
        PARTITION_KERNEL_STRING,
        boost::compute::system::default_context(),
        "-cl-std=CL2.0"
        );

    auto partition_kernel = program.create_kernel("partition_batched");

    auto min_val = 0.0f;
    auto max_val = 5000.0f;
    auto local_size = 256u;

    std::vector<data_t> pivots{ 1000.0f, 500.0f, 4000.0f, 0.0f, 5000.0f };


    auto checker = [&](unsigned single_batch_size, unsigned batches_count, unsigned groups_per_batch)
    {    
        std::vector<data_t> input = generate_uniformly_distributed_vec(single_batch_size * batches_count, min_val, max_val);
        std::vector<data_t> host_output(input.size());
        std::vector<parent_segment> host_parents(batches_count);
        for (auto i = 0u; i < batches_count; ++i)
        {
            host_parents[i].current_smaller_than_pivot_start_idx = i * single_batch_size;
            host_parents[i].current_greater_than_pivot_end_idx = (i + 1) * single_batch_size;
            host_parents[i].start_segment_global_idx = i * groups_per_batch;
        }


        bc::vector<data_t> src(input.begin(), input.end(), queue);
        bc::vector<data_t> dst(host_output.begin(), host_output.end(), queue);
        bc::vector<parent_segment> parents(1, queue.get_context());


        partition_kernel.set_arg(0, src);
        partition_kernel.set_arg(1, dst);
        partition_kernel.set_arg(2, parents);
        partition_kernel.set_arg(3, static_cast<unsigned>(input.size()));

        auto global_size = batches_count * groups_per_batch * local_size;

        for (auto pivot : pivots)
        {
            bc::copy(host_parents.begin(), host_parents.end(), parents.begin());

            partition_kernel.set_arg(4, pivot);
            queue.enqueue_1d_range_kernel(partition_kernel, 4, global_size, local_size);

            bc::copy(dst.begin(), dst.end(), host_output.begin(), queue);
     
            for (auto i = 0u; i < batches_count; ++i)
            {
                auto is_partition = is_partitioned(host_output.begin() + i * single_batch_size, host_output.begin() + (i + 1) * single_batch_size, [=](auto val) {return val < pivot; });
                REQUIRE(is_partition);
            }
        }
    };

    SECTION("single batch, one group per batch")
    {
        checker(local_size - 13, 1, 1);
        checker(local_size + 13, 1, 1);
        checker(local_size, 1, 1);
    }

    SECTION("single batch, multiple group per batch")
    {
        checker(local_size * 4 - 13, 1, 2);
        checker(local_size * 4 + 13, 1, 3);
        checker(local_size * 4, 1, 2);
    }
    SECTION("multiple batches, one group per batch")
    {
        checker(local_size * 3 - 13, 1, 1);
        checker(local_size * 3 + 13, 1, 1);
        checker(local_size * 6, 1, 1);
    }
    SECTION("multiple batches, multiple group per batch")
    {
        checker(local_size * 3 - 13, 1, 2);
        checker(local_size * 3 + 13, 1, 3);
        checker(local_size * 6, 1, 3);
    }
}
