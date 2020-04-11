#include "pch.hpp"
#include "catch.hpp"
#include "tests.utils.hpp"
#include "sorting_kernels.hpp"

namespace bc = boost::compute;


TEST_CASE("gpu partition million elements", "[algo] [partition] [fail1]")
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
    auto local_size = 2u;
    auto input_size = 3000u;

    std::vector<data_t> pivots{ 4000.0f, 500.0f, 1000.0f, 0.0f, 5000.0f };

    std::vector<data_t> input1 = generate_uniformly_distributed_vec(input_size, min_val, max_val);
    std::vector<data_t> input2 = generate_uniformly_distributed_vec(input_size, min_val, max_val);
    std::vector<data_t> input3 = generate_uniformly_distributed_vec(input_size, min_val, max_val);



    auto checker = [&](std::vector<data_t>& input, unsigned groups_per_chunk)
    {
        auto max_groups = static_cast<unsigned>(input.size()) / local_size + ((input.size() % local_size) > 0);
        auto groups = max_groups / groups_per_chunk;
        auto global_size = groups * local_size; // single chunk single work group
        std::vector<data_t> host_output(input.size());
        partition_segment segment{};
        segment.global_start_idx = 0;
        segment.global_end_idx = static_cast<unsigned>(input.size());


        partition_segment_result segment_result{};
        segment_result.smaller_than_pivot_upper = segment.global_start_idx;
        segment_result.greater_than_pivot_lower = segment.global_end_idx;
        segment_result.chunks_count_per_segment = groups;
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
            partition_kernel.set_arg(2, sizeof(partition_segment), &segment);
            queue.enqueue_1d_range_kernel(partition_kernel, 4, global_size, local_size);

            bc::copy(dst.begin(), dst.end(), host_output.begin(), queue);

            bc::copy(dst.begin(), dst.end(), host_output.begin(), queue);

            auto is_partition = is_partitioned(host_output.begin(), host_output.end(), [=](auto val) {return val < pivot; });


            auto greater_than_start_itr = std::partition(input.begin(), input.end(), [=](auto val) {return val < pivot; });
            auto greater_than_start_idx = std::distance(input.begin(), greater_than_start_itr);

            partition_segment_result result = device_result[0];

            REQUIRE(is_partition);

            REQUIRE(result.greater_than_pivot_lower == greater_than_start_idx);
        }
    };

    checker(input1, 1);
    //checker(input2, 1);
    //checker(input3, 1);

    //std::this_thread::sleep_for(std::chrono::milliseconds(5));

    //checker(input1, 2);
    //checker(input2, 2);
    //checker(input3, 2);

    //std::this_thread::sleep_for(std::chrono::milliseconds(5));

    //checker(input1, 4);
    //checker(input2, 4);
    //checker(input3, 4);
}


TEST_CASE("gpu partition - batched - 60 groups of 32K elements", "[algo] [partition] [fail2]")
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
    constexpr unsigned local_size = 256u;

    std::vector<data_t> pivots{ 1000.0f/*, 500.0f, 4000.0f, 0.0f, 5000.0f*/ };


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
            //host_segments[i].start_chunk_global_idx = i * groups_per_batch;
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
                host_segments_results[i].chunks_count_per_segment = groups_per_batch;
            }

            bc::copy(host_segments.begin(), host_segments.end(), segments.begin());
            bc::copy(host_segments_results.begin(), host_segments_results.end(), results.begin());
            queue.enqueue_1d_range_kernel(partition_kernel, 4, global_size, local_size);

            bc::copy(dst.begin(), dst.end(), host_output.begin(), queue);

            auto input_copy = input;

            for (auto i = 0u; i < batches_count; ++i)
            {
                auto is_partition = is_partitioned(host_output.begin() + i * single_batch_size, host_output.begin() + (i + 1) * single_batch_size, [=](auto val) {return val < pivot; });

                auto beg = input_copy.begin() + i * single_batch_size;
                auto end = input_copy.begin() + (i + 1) * single_batch_size;
                auto greater_than_start_itr = std::partition(beg, end, [=](auto val) {return val < pivot; });
                auto greater_than_start_idx = std::distance(input_copy.begin(), greater_than_start_itr);

                partition_segment_result result = results[i];


                REQUIRE(is_partition);
                REQUIRE(result.greater_than_pivot_lower == greater_than_start_idx);


         


            }


        }
    };

    constexpr unsigned single_batch_size = 1024 * 32;
    constexpr unsigned batches_count = 60;
    constexpr unsigned max_groups_per_batch = single_batch_size / local_size;

    checker(single_batch_size, batches_count, max_groups_per_batch);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    //checker(single_batch_size, batches_count, max_groups_per_batch / 2);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    //checker(single_batch_size, batches_count, max_groups_per_batch / 4);

}


