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


typedef struct partition_segment
{
    unsigned current_smaller_than_pivot_start_idx; // for allocation
    unsigned current_greater_than_pivot_end_idx;
    unsigned start_segment_global_idx;
} partition_segment;