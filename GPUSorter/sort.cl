R""(
//#include "partition.cl"

typedef float data_t;
typedef uint idx_t;

data_t median3(data_t x1, data_t x2, data_t x3) 
{
    return min(max(x1, x2), x3);
}



// the recursion_dispatcher kernel is enqueued after each partition
// its job is to dispatch 2 additional partitions in a recursive manner
kernel void recursion_dispatcher(
    global data_t* src,
    global data_t* dst,
    global partition_segment* segments,
    global partition_segment_chunk* chunks,
    global partition_segment_result* results
    )
{
     partition_segment_result result = results[0];

     partition_segment left_segment;
     left_segment.global_start_idx = segment.global_start_idx;
     left_segment.global_end_idx = result.smaller_than_pivot_upper; // +1 ? 
     
     partition_segment right_segment;
     right_segment.global_start_idx = result.greater_than_pivot_lower;
     right_segment.global_end_idx = segment.global_end_idx;
}

typedef struct partition_segment_chunk_ex
{
    idx_t segment_idx;
    partition_segment_chunk chunk;
} partition_segment_chunk_ex;






typedef struct bitonic_segment
{
    idx_t global_start_idx;
    idx_t global_end_idx;
} bitonic_segment;



kernel void sort(
    global data_t* src,
    global data_t* dst,
    global partition_segment* segments,
    global partition_segment_chunk_ex* chunks,
    global partition_segment_result* results,
    global partition_segment* dst_segments,
    global partition_segment_result* dst_results,
    global bitonic_segment* bitonic_segments,
    idx_t segments_count
    )
{
    idx_t group_idx = get_group_id(0);
    partition_segment_chunk_ex chunk_ex = chunks[group_idx];
    partition_segment segment = segments[chunk_ex.segment_idx];
    partition_segment_result* result = results + chunk_ex.segment_idx;

    local idx_t smaller_than_pivot_global_offset;
    local idx_t greater_than_pivot_global_offset;  
    local idx_t last_group_counter;

    local idx_t chunks_allocate_idx;
    local idx_t bitonic_chunks_allocate_idx;
    if (local_idx == 0)
    {
        chunks_allocate_idx = 0;
        bitonic_segments_allocate_idx = 0;
        // no need for a barrier, there will be plenty before these are used
    }

    segment_partition(
        src,
        dst,
        segment,
        chunk_ex.chunk,
        result,
        &smaller_than_pivot_global_offset,
        &greater_than_pivot_global_offset,
        &last_group_counter
        );


    // partitions are done, we now examine the results and recursively dispatch the same kernel
    // for all large enough result segments

    if (last_group_counter == 1)
    {
        for (idx_t i = local_idx; i < segments_count; i+= group_size)
        {   
            partition_segment segment = segments[i],
            partition_segment_result current_result = result[i];
            partition_segment left_segment;
            left_segment.global_start_idx = segment.global_start_idx;
            left_segment.global_end_idx = result.smaller_than_pivot_upper; // +1 ? 
            dst_segments[i*2] = left_segment;
            partition_segment right_segment;
            right_segment.global_start_idx = result.greater_than_pivot_lower;
            right_segment.global_end_idx = segment.global_end_idx;
            dst_segments[i*2 + 1] = right_segment;

            idx_t total_left = result.smaller_than_pivot_upper - segment.global_start_idx; 
            idx_t total_right = right_segment.global_end_idx - right_segment.global_start_idx; 

            if (total_left > 512)
            {                
                idx_t chunks_count = total_left / (256 * 4);
                // allocate and compute chunks
                
            }
            else if (total_left > 0)
            {
                // bitonic sort this segment inplace, or to a different buffer if needed
                idx_t bitonic_idx = atomic_inc(bitonic_segments_allocate_idx);
                bitonic_segment left_segment;
                left_segment.global_start_idx = segment.global_start_idx;
                left_segment.global_end_idx = result.smaller_than_pivot_upper; // +1 ?    
            }
        }

        if (local_idx == 0)
        {
            if (bitonic_segments_allocate_idx > 0)
            {
                // dispatch bitonic sort to same destination
            }
            if (chunks_allocate_idx > 0)
            {
                // dispatch this kernel again using double amount of segments
                queue_t q = get_default_queue();
                enqueue_kernel(
                    q,
                    CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
                    ndrange_1D(1),
                    ^{ relauncher_kernel(d, dn, blocks, parents, result, work, done, done_size, MAXSEQ, num_workgroups); }
                ); 
            }           
        }
    }

}





    //idx_t global_size = get_global_size();

    //// select pivot
    //partition_segment segment = segments[0];
    //idx_t groups_count = get_num_groups(0);
    //idx_t elements_per_group = (segment.global_end_idx - segment.global_start_idx) / groups_count;


    //partition_segment_chunk chunk;
    //chunk.start = elements_per_group * group_idx;
    //chunk.end = (group_idx == (groups_count - 1)) ? segment.global_end_idx : elements_per_group * (group_idx + 1);
    //
    //idx_t local_idx = get_local_id(0);
    //segment.pivot = src[(segment.global_start_idx + segment.global_end_idx) / 2];

    //local idx_t smaller_than_pivot_global_offset;
    //local idx_t greater_than_pivot_global_offset;  
    //local idx_t last_group_counter;




)""