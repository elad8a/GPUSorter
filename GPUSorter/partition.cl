R""(

typedef float data_t;
typedef uint idx_t;

typedef struct partition_segment
{
    idx_t current_smaller_than_pivot_start_idx; // for allocation
    idx_t current_greater_than_pivot_end_idx; 
    idx_t start_segment_global_idx;
} partition_segment;

typedef struct partition_segment_chunk
{
    idx_t start; // inclusive, global start of the partition chunk
    idx_t end; // exclusive
    idx_t pivot;
    idx_t parent_segment_idx;
} partition_segment_chunk;

void segment_partition(
    global data_t* src, 
    global data_t* dst,
    partition_segment_chunk chunk,
    global partition_segment* parent_segments,
    local idx_t* smaller_than_pivot_global_offset,
    local idx_t* greater_than_pivot_global_offset
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
        smaller_than_pivot_private_count += (val < chunk.pivot);
        greater_than_pivot_private_count += (val > chunk.pivot);
    }

    idx_t smaller_than_pivot_exclusive_cumulative_count = work_group_scan_exclusive_add(smaller_than_pivot_private_count);
    idx_t greater_than_pivot_exclusive_cumulative_count = work_group_scan_exclusive_add(greater_than_pivot_private_count);

    global partition_segment* parent =  parent_segments + chunk.parent_segment_idx;
    
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
    for (idx_t i = chunk.start + local_idx; i < chunk.end; i += group_size)
    {
        data_t val = src[i];
        // increment counts
        if (val < chunk.pivot)
        {
            dst[smaller_private_begin_global_idx++] = val;
        }
        else if (val > chunk.pivot)
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
                dst[global_start_idx + offset_idx] = chunk.pivot /*global_start_idx + offset_idx */ ;
            }  
        }
    }
}



__kernel void partition(    
    global data_t* src, 
    global data_t* dst,
    global partition_segment* parent,
    idx_t count,
    data_t pivot
    )
{    
    local idx_t smaller_than_pivot_global_offset;
    local idx_t greater_than_pivot_global_offset;

    partition_segment_chunk chunk;
    chunk.start = 0; // inclusive, global start of the partition chunk
    chunk.end = count; // exclusive
    chunk.pivot = pivot;
    chunk.parent_segment_idx = 0;

    segment_partition(
        src,
        dst,
        chunk,
        parent,
        &smaller_than_pivot_global_offset,
        &greater_than_pivot_global_offset
        );

}



__kernel void partition_batched(    
    global data_t* src, 
    global data_t* dst,
    global partition_segment* parents,
    idx_t single_batch_size,
    idx_t batches_count,
    data_t pivot
    )
{    
    local idx_t smaller_than_pivot_global_offset;
    local idx_t greater_than_pivot_global_offset;

    idx_t groups_count = get_num_groups(0);
    idx_t groups_per_batch = groups_count / batches_count; // assumes groups_count = K * batches_count where K is a positive integral
    idx_t elements_per_group = single_batch_size / groups_per_batch;
    idx_t elements_per_group_remainder = single_batch_size % groups_per_batch;

    const idx_t batch_idx = get_group_id(0) / groups_per_batch;
    const idx_t idx_within_batch = get_group_id(0) % groups_per_batch;


    partition_segment_chunk chunk;
    chunk.start = batch_idx * single_batch_size + idx_within_batch * elements_per_group; // inclusive, global start of the partition chunk
    chunk.end = chunk.start + elements_per_group; // exclusive
    if ((idx_within_batch + 1) == groups_per_batch)
    {
        chunk.end += elements_per_group_remainder;
    }
    
    chunk.pivot = pivot;
    chunk.parent_segment_idx = batch_idx;

    segment_partition(
        src,
        dst,
        chunk,
        parents,
        &smaller_than_pivot_global_offset,
        &greater_than_pivot_global_offset
        );

}
)""