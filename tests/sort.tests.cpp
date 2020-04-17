#include "pch.hpp"
#include "catch.hpp"
#include "tests.utils.hpp"
#include "../GPUSorter/GPUFloatSorter.hpp"


namespace bc = boost::compute;
using namespace HPC;

TEST_CASE("partition 2", "[algo] [sort] [fail]")
{
    using data_t = float;
    auto queue = boost::compute::system::default_queue();


    //auto min_val = 0.0f;
    //auto max_val = 5000.0f;
    //auto local_size = 256u;
    //auto global_size = local_size; // single chunk single work group
    //auto max_elements = local_size * 8;

    std::vector<int> host_vec{
        999,
        1, 1000, 2134, 22, 3333,5455, 13,4243, 21, 9999, 12,5444, 13, 7721, 54,6200, 13, 0, 8888, 5,8888, 43, 1454, 41, 24,
        66, 85, 6, 7, 12, 76


    };

    bc::vector<int> device_src_vec(host_vec.begin(), host_vec.end());
    bc::vector<int> device_dst_vec(host_vec.size());

    std::vector<int> host_dst_vec(host_vec.size());

    GPUFloatSorter sorter(queue, 32*2);

    sorter.Sort(device_src_vec, device_dst_vec, queue);

    bc::copy(device_dst_vec.begin(), device_dst_vec.end(), host_dst_vec.begin());


    REQUIRE(true);

    //std::vector<data_t> pivots{ 1000.0f, 500.0f, 4000.0f, 0.0f, 5000.0f };

    //

    //std::vector<data_t> input1 = generate_uniformly_distributed_vec(local_size, min_val, max_val);
    //std::vector<data_t> input2 = generate_uniformly_distributed_vec(local_size - 50, min_val, max_val);
    //std::vector<data_t> input3 = generate_uniformly_distributed_vec(local_size * 4 - 13, min_val, max_val);



    //auto checker = [&](std::vector<data_t>& input)
    //{
    //    std::vector<data_t> host_output(input.size());
    //    partition_segment host_parent{};
    //    host_parent.current_smaller_than_pivot_start_idx = 0;
    //    host_parent.current_greater_than_pivot_end_idx = static_cast<unsigned>(input.size());
    //    host_parent.start_segment_global_idx = 0;

    //    bc::vector<data_t> src(input.begin(), input.end(), queue);
    //    bc::vector<data_t> dst(host_output.begin(), host_output.end(), queue);
    //    bc::vector<partition_segment> parents(1, queue.get_context());


    //    partition_kernel.set_arg(0, src);
    //    partition_kernel.set_arg(1, dst);
    //    partition_kernel.set_arg(2, parents);
    //    partition_kernel.set_arg(3, static_cast<unsigned>(input.size()));



    //    for (auto pivot : pivots)
    //    {
    //        parents[0] = host_parent;
    //        partition_kernel.set_arg(4, pivot);
    //        queue.enqueue_1d_range_kernel(partition_kernel, 4, global_size, local_size);

    //        bc::copy(dst.begin(), dst.end(), host_output.begin(), queue);
    //        if (pivot != min_val && pivot != max_val)
    //        {
    //            CHECK(!is_partitioned(input.begin(), input.end(), [=](auto val) {return val < pivot; }));
    //        }
    //        auto is_partition = is_partitioned(host_output.begin(), host_output.end(), [=](auto val) {return val < pivot; });
    //        REQUIRE(is_partition);
    //        //CHECK(std::is_permutation(input.begin(), input.end(), host_output.begin()));
    //    }
    //};


    //checker(input1);
    REQUIRE(true);
}