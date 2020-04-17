#pragma once


namespace HPC
{
    using data_t = int;
    using idx_t = unsigned;

    struct partition_segment
    {
        // static data
        data_t pivot;
        idx_t global_start_idx;
        idx_t global_end_idx;
    };

    struct partition_segment_result
    {
        // data changed by kernels
        idx_t smaller_than_pivot_upper; // for allocation
        idx_t greater_than_pivot_lower;
        idx_t chunks_count_per_segment; // used internally by kernel
    };

    struct partition_segment_chunk
    {
        idx_t start; // inclusive, global start of the partition chunk
        idx_t end; // exclusive
    };

    struct partition_segment_chunk_ex
    {
        idx_t segment_idx;
        partition_segment_chunk chunk;
    };

    struct bitonic_segment
    {
        idx_t global_start_idx;
        idx_t global_end_idx;
    };


}