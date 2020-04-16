R""(
#ifndef TARGET_TYPES
#define TARGET_TYPES
typedef float data_t;
typedef uint idx_t;
#endif

// a partion segment represent a single partition
// performed on a segment of the data array
// each partition segment is further divided into chunks
// each chunk is processed by a single work group
typedef struct partition_segment
{
    // static data
    data_t pivot;
    idx_t global_start_idx;
    idx_t global_end_idx;
} partition_segment;

// partition_segment_result 
// contains a single partition segment result.
// it contains 2 elements:
// - the upper end index of the 'smaller than pivot' elements
// - the lower end in of the 'greater than pivot' elements 
// these elements are initialized to the initial ends of the
// partition segment and are used by the kernels for allocation 
// purposes as well as storing results
typedef struct partition_segment_result
{
    // data changed by kernels
    idx_t smaller_than_pivot_upper; // for allocation
    idx_t greater_than_pivot_lower; 
    idx_t chunks_count_per_segment;   
} partition_segment_result;

typedef struct partition_segment_chunk
{
    idx_t start; // inclusive, global start of the partition chunk
    idx_t end; // exclusive
    //idx_t parent_segment_idx; // no need?
} partition_segment_chunk;

void segment_partition(
    global data_t* src, 
    global data_t* dst,
    partition_segment segment,
    partition_segment_chunk chunk,
    global partition_segment_result* result,
    local idx_t* smaller_than_pivot_global_offset,
    local idx_t* greater_than_pivot_global_offset,
    local idx_t* last_group_counter
    )
{
    const idx_t local_idx = get_local_id(0);
    const idx_t group_size = get_local_size(0);

    idx_t smaller_than_pivot_private_count = 0;
    idx_t greater_than_pivot_private_count = 0;


    // stage 1: allocation phase
    for (idx_t i = chunk.start + local_idx; i < chunk.end; i += group_size)
    {
        data_t val = src[i];
        smaller_than_pivot_private_count += (val < segment.pivot);
        greater_than_pivot_private_count += (val > segment.pivot);
    }

    idx_t packed = (greater_than_pivot_private_count << 16) | smaller_than_pivot_private_count; // a genius move no less
    idx_t cumulative_packed = work_group_scan_exclusive_add(packed); 
    idx_t smaller_than_pivot_exclusive_cumulative_count = cumulative_packed & 0x0000FFFF;
    idx_t greater_than_pivot_exclusive_cumulative_count = cumulative_packed >> 16;
    //idx_t smaller_than_pivot_exclusive_cumulative_count = work_group_scan_exclusive_add(smaller_than_pivot_private_count);
    //idx_t greater_than_pivot_exclusive_cumulative_count = work_group_scan_exclusive_add(greater_than_pivot_private_count);

    //volatile global idx_t* p_smaller_than_pivot_upper = &result->smaller_than_pivot_upper;
    //volatile global idx_t* p_greater_than_pivot_lower = &result->greater_than_pivot_lower;

    if (local_idx == (group_size - 1)) 
    { 
        *last_group_counter = 999; // magic number, does not make a difference as long as it is not 1
        // last partition_work item has the total counts minus last element
        idx_t smaller_than_count = smaller_than_pivot_exclusive_cumulative_count + smaller_than_pivot_private_count;
        idx_t greater_than_count = greater_than_pivot_exclusive_cumulative_count + greater_than_pivot_private_count;

        // atomic increment allocates memory to write to.
        *smaller_than_pivot_global_offset = atomic_add(
            &result->smaller_than_pivot_upper,
            smaller_than_count
            );

        // atomic is necessary since multiple groups access this
        *greater_than_pivot_global_offset = atomic_sub(
            &result->greater_than_pivot_lower,
            greater_than_count
            ) - greater_than_count;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    idx_t smaller_private_begin_global_idx = *smaller_than_pivot_global_offset + smaller_than_pivot_exclusive_cumulative_count;
    idx_t greater_private_begin_global_idx = *greater_than_pivot_global_offset + greater_than_pivot_exclusive_cumulative_count;

    // go through data again writing elements to their correct position
    for (idx_t i = chunk.start + local_idx; i < chunk.end; i += group_size)
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

    idx_t total_pivots = 0;
    if (local_idx == 0)
    {
         *last_group_counter = atomic_dec(&result->chunks_count_per_segment);
    }  

    barrier(CLK_LOCAL_MEM_FENCE);
   
    if (*last_group_counter == 1) 
    {
        partition_segment_result current_result = *result;
        idx_t global_start_idx = current_result.smaller_than_pivot_upper;
        int total_pivots = current_result.greater_than_pivot_lower - global_start_idx;
        if (total_pivots > 0)
        {
            for (uint offset_idx = local_idx; offset_idx < total_pivots; offset_idx += group_size)
            {
                //dst[global_start_idx + offset_idx] = 123;
                dst[global_start_idx + offset_idx] = segment.pivot /*global_start_idx + offset_idx */;
            }  
        }
    }
}

__kernel void partition(    
    global data_t* src, 
    global data_t* dst,
    partition_segment segment,
    global partition_segment_result* result
    )
{    
    local idx_t smaller_than_pivot_global_offset;
    local idx_t greater_than_pivot_global_offset;  
    local idx_t last_group_counter;

    idx_t groups_count = get_num_groups(0);
    idx_t elements_per_group = (segment.global_end_idx - segment.global_start_idx) / groups_count;
    idx_t group_idx = get_group_id(0);

    partition_segment_chunk chunk;
    chunk.start = elements_per_group * group_idx; // inclusive, global start of the partition chunk
    chunk.end = (group_idx == (groups_count - 1)) ? segment.global_end_idx : elements_per_group * (group_idx + 1);

    segment_partition(
        src,
        dst,
        segment,
        chunk,
        result,
        &smaller_than_pivot_global_offset,
        &greater_than_pivot_global_offset,
        &last_group_counter
        );

}



__kernel void partition_batched(    
    global data_t* src, 
    global data_t* dst,
    __constant partition_segment* segments, 
    global partition_segment_result* results,
    idx_t single_batch_size,
    idx_t batches_count
    )
{    
    local idx_t smaller_than_pivot_global_offset;
    local idx_t greater_than_pivot_global_offset;
    local idx_t last_group_counter;

    idx_t groups_count = get_num_groups(0);
    idx_t groups_per_batch = groups_count / batches_count; // assumes groups_count = K * batches_count where K is a positive integral
    idx_t elements_per_group = single_batch_size / groups_per_batch;

    const idx_t batch_idx = get_group_id(0) / groups_per_batch;
    const idx_t idx_within_batch = get_group_id(0) % groups_per_batch;

    partition_segment segment = segments[batch_idx];


    partition_segment_chunk chunk;
    chunk.start = batch_idx * single_batch_size + idx_within_batch * elements_per_group; // inclusive, global start of the partition chunk
    chunk.end = ((idx_within_batch + 1) == groups_per_batch) ? segment.global_end_idx : chunk.start + elements_per_group; // exclusive

    
    segment_partition(
        src,
        dst,
        segment,
        chunk,
        results + batch_idx,
        &smaller_than_pivot_global_offset,
        &greater_than_pivot_global_offset,
        &last_group_counter
        );

}
)""