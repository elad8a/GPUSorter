R""(
//#include "partition.cl"

typedef float data_t;
typedef uint idx_t;

data_t median3(data_t x1, data_t x2, data_t x3) 
{
    return min(max(x1, x2), x3);
}

typedef struct work_descriptor
{
    uint start;
    uint end;
    uint pivot;
    uint direction; // 1 - read from data_1 write to data_2, 0 - the other way. 42 - do not write, nothing to sort
} work_descriptor_t;


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


// the recursive_partition kernel performs a partition and then
// enqueues the recursion_dispatcher so it will enqueue additional
// partitions as necessary
kernel void recursive_partition(
    global data_t* src,
    global data_t* dst,
    global partition_segment* segments,
    global partition_segment_chunk* chunks,
    global partition_segment_result* results
    )
{

}

kernel void sort(
    global data_t* src,
    global data_t* dst,
    global partition_segment* segments,
    global partition_segment_chunk* chunks,
    global partition_segment_result* results
    )
{
    // select pivot
    partition_segment segment = segments[0];
    idx_t groups_count = get_num_groups(0);
    idx_t elements_per_group = (segment.global_end_idx - segment.global_start_idx) / groups_count;
    idx_t group_idx = get_group_id(0);

    partition_segment_chunk chunk;
    chunk.start = elements_per_group * group_idx;
    chunk.end = (group_idx == (groups_count - 1)) ? segment.global_end_idx : elements_per_group * (group_idx + 1);
    
    idx_t local_idx = get_local_id(0);
    segment.pivot = src[(segment.global_start_idx + segment.global_end_idx) / 2];

    local idx_t smaller_than_pivot_global_offset;
    local idx_t greater_than_pivot_global_offset;  
    local idx_t last_group_counter;

    segment_partition(
        src,
        dst,
        segment,
        chunk,
        results + 0,
        &smaller_than_pivot_global_offset,
        &greater_than_pivot_global_offset,
        &last_group_counter
        );

    // first partition is done, we now launch the wrapper
    if (get_global_id(0) == 0)
    {
        queue_t q = get_default_queue();
        enqueue_kernel(
            q,
            CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
            ndrange_1D(1),
            ^{ relauncher_kernel(d, dn, blocks, parents, result, work, done, done_size, MAXSEQ, num_workgroups); }
        ); 

        
    }
}

)""