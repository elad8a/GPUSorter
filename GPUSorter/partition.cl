R""(
typedef float data_t;
typedef uint idx_t;

#define QUICKSORT_CHUNK_MIN_SIZE        1 // FOR now, keep partitioning until done 
#define EMPTY_RECORD                   42
#define GQSORT_LOCAL_WORKGROUP_SIZE   256
#define LQSORT_LOCAL_WORKGROUP_SIZE   256

data_t median(data_t x1, data_t x2, data_t x3) 
{
    if (x1 < x2) 
    {
        if (x2 < x3) 
        {
            return x2;
        }
        else 
        {
            if (x1 < x3) 
            {
                return x3;
            }
            else 
            {
                return x1;
            }
        }
    }
    else 
    { // x1 >= x2
        if (x1 < x3) 
        {
            return x1;
        }
        else 
        { // x1 >= x3
            if (x2 < x3) 
            {
                return x2;
            }
            else 
            {
                return x3;
            }
        }
    }
}

typedef struct parent_segment
{
    idx_t current_smaller_than_pivot_start_idx; // for allocation
    idx_t current_greater_than_pivot_end_idx; 
    idx_t start_segment_global_idx;
} parent_segment;

typedef struct partition_segment
{
    idx_t start; // inclusive, global start of the partition segment
    idx_t end; // exclusive
    idx_t pivot;
    idx_t parent_segment_idx;
} partition_segment;

void segment_partition(
    global data_t* src, 
    global data_t* dst,
    partition_segment segment,
    global parent_segment* parent_segments,
    local idx_t* smaller_than_pivot_global_offset,
    local idx_t* greater_than_pivot_global_offset
    )
{

    const idx_t local_idx = get_local_id(0);
    const idx_t group_size = get_local_size(0);


    idx_t smaller_than_pivot_private_count = 0;
    idx_t greater_than_pivot_private_count = 0;


    // stage 1: allocation phase
    for (idx_t i = segment.start + local_idx; i < segment.end; i += group_size)
    {
        data_t val = src[i];
        smaller_than_pivot_private_count += (val < segment.pivot);
        greater_than_pivot_private_count += (val > segment.pivot);
    }

    idx_t smaller_than_pivot_exclusive_cumulative_count = work_group_scan_exclusive_add(smaller_than_pivot_private_count);
    idx_t greater_than_pivot_exclusive_cumulative_count = work_group_scan_exclusive_add(greater_than_pivot_private_count);

    global parent_segment* parent =  parent_segments + segment.parent_segment_idx;
    
    if (local_idx == (group_size - 1)) 
    { 
        // last partition_work item has the total counts minus last element
        idx_t smaller_than_count = smaller_than_pivot_exclusive_cumulative_count + smaller_than_pivot_private_count;
        idx_t greater_than_count = greater_than_pivot_exclusive_cumulative_count + greater_than_pivot_private_count;

        // atomic increment allocates memory to write to.
        *smaller_than_pivot_global_offset = atomic_add(
            &parent->current_smaller_than_pivot_start_idx,
            smaller_than_count
            );

        // atomic is necessary since multiple groups access this
        *greater_than_pivot_global_offset = atomic_sub(
            &parent->current_greater_than_pivot_end_idx,
            greater_than_count
            ) - greater_than_count;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    idx_t smaller_private_begin_global_idx = *smaller_than_pivot_global_offset + smaller_than_pivot_exclusive_cumulative_count;
    idx_t greater_private_begin_global_idx = *greater_than_pivot_global_offset + greater_than_pivot_exclusive_cumulative_count;

    // go through data again writing elements to their correct position
    for (idx_t i = segment.start + local_idx; i < segment.end; i += group_size)
    {
        data_t val = src[i];
        // increment counts
        if (val < segment.pivot)
        {
            dst[smaller_private_begin_global_idx++] = val;
        }
        else if (val > segment.pivot)
        {
            dst[greater_private_begin_global_idx++] = val;
        }
    }

    // wait for all threads to finish
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE); // necessary? yes, all threads need to finish partitioning?

    const idx_t group_idx = get_group_id(0);
    if (group_idx == parent->start_segment_global_idx)
    {
        uint global_start_idx = parent->current_smaller_than_pivot_start_idx;
        uint global_end_idx = parent->current_greater_than_pivot_end_idx;

        int total_pivots = global_end_idx - global_start_idx;
        if (total_pivots > 0)
        {
            for (uint offset_idx = local_idx; offset_idx < total_pivots; offset_idx += group_size)
            {
                dst[global_start_idx + offset_idx] = segment.pivot /*global_start_idx + offset_idx */ ;
            }  
        }
    }
}

__kernel void partition(    
    global data_t* src, 
    global data_t* dst,
    global parent_segment* parent,
    idx_t count,
    data_t pivot
    )
{    
    local idx_t smaller_than_pivot_global_offset;
    local idx_t greater_than_pivot_global_offset;

    const idx_t segment_idx = get_group_id(0);

    partition_segment segment;
    segment.start = 0; // inclusive, global start of the partition segment
    segment.end = count; // exclusive
    segment.pivot = pivot;
    segment.parent_segment_idx = 0;

    segment_partition(
        src,
        dst,
        segment,
        parent,
        &smaller_than_pivot_global_offset,
        &greater_than_pivot_global_offset
        );

}
)""
