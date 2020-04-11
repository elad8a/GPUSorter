#include "pch.hpp"
#include "catch.hpp"
#include "tests.utils.hpp"
#include "sorting_kernels.hpp"

namespace bc = boost::compute;

TEST_CASE("partition 2", "[algo] [sort] [fail]")
{
    using data_t = float;
    auto queue = boost::compute::system::default_queue();

    const char* strings[3]
    {
        PARTITION_KERNEL_STRING.c_str(),
        BITONIC_SORT_KERNEL_STRING.c_str(),
        QSORT_KERNEL_STRING.c_str()
    };

    std::size_t lengths[3]
    {
        PARTITION_KERNEL_STRING.size(),
        BITONIC_SORT_KERNEL_STRING.size(),
        QSORT_KERNEL_STRING.size()
    };

    cl_int err;

    auto prog = clCreateProgramWithSource(queue.get_context(), 2, strings, lengths, &err);
    auto program = bc::program(prog, false);
    program.build("-cl-std=CL2.0");


    auto partition_kernel = program.create_kernel("partition2");

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
        partition_segment host_parent{};
        host_parent.current_smaller_than_pivot_start_idx = 0;
        host_parent.current_greater_than_pivot_end_idx = static_cast<unsigned>(input.size());
        host_parent.start_segment_global_idx = 0;

        bc::vector<data_t> src(input.begin(), input.end(), queue);
        bc::vector<data_t> dst(host_output.begin(), host_output.end(), queue);
        bc::vector<partition_segment> parents(1, queue.get_context());


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


    checker(input1);
    REQUIRE(true);
}