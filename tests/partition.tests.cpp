#include "pch.hpp"
#include "catch.hpp"
#include "tests.utils.hpp"
#include "sorting_kernels.hpp"

namespace bc = boost::compute;


TEST_CASE("gpu partition", "[algo] [partition]")
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


    std::vector<data_t> pivots{ 1000.0f, 500.0f, 4000.0f, 0.0f, 5000.0f };

    std::vector<data_t> input1 = generate_uniformly_distributed_vec(local_size, min_val, max_val);
    std::vector<data_t> input2 = generate_uniformly_distributed_vec(local_size - 50, min_val, max_val);
    std::vector<data_t> input3 = generate_uniformly_distributed_vec(local_size * 4 - 13, min_val, max_val);

    

    auto checker = [&](std::vector<data_t>& input)
    {    
        auto max_groups = static_cast<unsigned>(input.size()) / local_size + ((input.size() % local_size) > 0);
        auto global_size = max_groups * local_size; // single chunk single work group
        std::vector<data_t> host_output(input.size());
        partition_segment segment{};
        segment.global_start_idx = 0;
        segment.global_end_idx = static_cast<unsigned>(input.size());
        segment.start_chunk_global_idx = 0;

        partition_segment_result segment_result{};
        segment_result.smaller_than_pivot_upper = segment.global_start_idx;
        segment_result.greater_than_pivot_lower = segment.global_end_idx;
        bc::vector<partition_segment_result> device_result;

        bc::vector<data_t> src(input.begin(), input.end(), queue);
        bc::vector<data_t> dst(input.size(), queue.get_context());
        partition_kernel.set_arg(0, src);
        partition_kernel.set_arg(1, dst);
        partition_kernel.set_arg(3, device_result);



        for (auto pivot : pivots)
        {
            segment.pivot = pivot;
            device_result[0] = segment_result;
            partition_kernel.set_arg(2,sizeof(partition_segment), &segment);
            queue.enqueue_1d_range_kernel(partition_kernel, 4, global_size, local_size);

            bc::copy(dst.begin(), dst.end(), host_output.begin(), queue);
            if (pivot != min_val && pivot != max_val)
            {
                CHECK(!is_partitioned(input.begin(), input.end(), [=](auto val) {return val < pivot; }));
            }
            auto is_partition = is_partitioned(host_output.begin(), host_output.end(), [=](auto val) {return val < pivot; });


            REQUIRE(is_partition);

            auto greater_than_start_itr = std::partition(input.begin(), input.end(), [=](auto val) {return val < pivot; });
            auto greater_than_start_idx = std::distance(input.begin(), greater_than_start_itr);

            partition_segment_result result = device_result[0];
            REQUIRE(result.greater_than_pivot_lower == greater_than_start_idx);
            //CHECK(std::is_permutation(input.begin(), input.end(), host_output.begin()));
        }
    };

    //SECTION("segment size == local_size")
    //{
    //    checker(input1);
    //}
    //SECTION("segment size < local_size")
    //{
    //    checker(input2);
    //}
    SECTION("segment size > local_size")
    {
        checker(input3);
    }


}


TEST_CASE("gpu partition - batched", "[algo] [partition]")
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
        std::vector<partition_segment> host_segments(batches_count);
        std::vector<partition_segment_result> host_segments_results(batches_count);
        for (auto i = 0u; i < batches_count; ++i)
        {
            host_segments[i].global_start_idx = i * single_batch_size;
            host_segments[i].global_end_idx = (i + 1) * single_batch_size;
            host_segments[i].start_chunk_global_idx = i * groups_per_batch;
        }


        bc::vector<data_t> src(input.begin(), input.end(), queue);
        bc::vector<data_t> dst(host_output.begin(), host_output.end(), queue);
        bc::vector<partition_segment> segments(host_segments.size(), queue.get_context());
        bc::vector<partition_segment_result> results(host_segments_results.size(), queue.get_context());
        partition_kernel.set_arg(0, src);
        partition_kernel.set_arg(1, dst);
        partition_kernel.set_arg(2, segments);
        partition_kernel.set_arg(3, results);
        partition_kernel.set_arg(4, single_batch_size);
        partition_kernel.set_arg(5, batches_count);

        auto global_size = batches_count * groups_per_batch * local_size;

        for (auto pivot : pivots)
        {
            for (auto i = 0u; i < batches_count; ++i)
            {
                host_segments[i].pivot = pivot;
                host_segments_results[i].smaller_than_pivot_upper = host_segments[i].global_start_idx;
                host_segments_results[i].greater_than_pivot_lower = host_segments[i].global_end_idx;
            }

            bc::copy(host_segments.begin(), host_segments.end(), segments.begin());
            bc::copy(host_segments_results.begin(), host_segments_results.end(), results.begin());
            queue.enqueue_1d_range_kernel(partition_kernel, 4, global_size, local_size);

            bc::copy(dst.begin(), dst.end(), host_output.begin(), queue);
     
            auto input_copy = input;

            for (auto i = 0u; i < batches_count; ++i)
            {
                auto is_partition = is_partitioned(host_output.begin() + i * single_batch_size, host_output.begin() + (i + 1) * single_batch_size, [=](auto val) {return val < pivot; });
                REQUIRE(is_partition);

                auto beg = input_copy.begin() + i * single_batch_size;
                auto end = input_copy.begin() + (i + 1) * single_batch_size;
                auto greater_than_start_itr = std::partition(beg, end, [=](auto val) {return val < pivot; });
                auto greater_than_start_idx = std::distance(input_copy.begin(), greater_than_start_itr);

                partition_segment_result result = results[i];
                REQUIRE(result.greater_than_pivot_lower == greater_than_start_idx);
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
        checker(local_size * 2, 1, 2);
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


