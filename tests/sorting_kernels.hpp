#pragma once
inline const std::string PARTITION_KERNEL_STRING =
#include "../GPUSorter/partition.cl"
"";
inline const std::string QSORT_KERNEL_STRING =
#include "../GPUSorter/qsort.cl"
"";

inline const std::string BITONIC_SORT_KERNEL_STRING =
#include "../GPUSorter/bitonic_sort.cl"
"";

using data_t = float;
using idx_t = unsigned;

typedef struct partition_segment
{
    // static data
    data_t pivot;
    idx_t global_start_idx;
    idx_t global_end_idx;
    idx_t start_chunk_global_idx;
} partition_segment;

typedef struct partition_segment_result
{
    // data changed by kernels
    idx_t smaller_than_pivot_upper; // for allocation
    idx_t greater_than_pivot_lower;
} partition_segment_result;