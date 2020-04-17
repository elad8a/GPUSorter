R""(
//#include "partition.cl"

#ifndef TARGET_TYPE_GUARD
#define TARGET_TYPE_GUARD
typedef uint idx_t;
#ifdef TARGET_TYPE
typedef TARGET_TYPE data_t;
#else
typedef float data_t;
#endif
#endif

#ifndef MAX_GROUP_SIZE
#define MAX_GROUP_SIZE 256
#endif

#define ALTERNATIVE_SORT_THRESHOLD (MAX_GROUP_SIZE * 2)

#ifndef PARTITION_ELEMENTS_PER_WORKGROUP
#define PARTITION_ELEMENTS_PER_WORKGROUP (MAX_GROUP_SIZE * 4)
#endif


data_t median3(data_t x1, data_t x2, data_t x3) 
{
    return min(max(x1, x2), x3);
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

kernel void sort3(
    global data_t* src,
    global data_t* dst,
    global partition_segment* segments,
    global partition_segment_chunk_ex* chunks,
    global partition_segment_result* results,
    global partition_segment* dst_segments,
    global partition_segment_result* dst_results,
    global bitonic_segment* bitonic_segments,
    idx_t bitonic_segments_count,
    idx_t segments_count    
    )
{
    if (get_global_id(0) == 0)
    {
        printf("hello 3");
    }

}


kernel void sort2(
    global data_t* src,
    global data_t* dst,
    global partition_segment* segments,
    global partition_segment_chunk_ex* chunks,
    global partition_segment_result* results,
    global partition_segment* dst_segments,
    global partition_segment_result* dst_results,
    global bitonic_segment* bitonic_segments,
    idx_t bitonic_segments_count,
    idx_t segments_count    
    )
{
    if (get_global_id(0) == 0)
    {
        printf("hello 2");

        queue_t q = get_default_queue();
        enqueue_kernel(
            q,
            CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
            ndrange_1D(1),
            ^{ 
            sort3(
                src,
                dst,
                segments,
                chunks,
                results,
                dst_segments,
                dst_results,                        
                bitonic_segments,
                0,
                0    
                ); 
            }
        
        ); 
    }

}

kernel void sort(
    global data_t* src,
    global data_t* dst,
    global partition_segment* segments,
    global partition_segment_chunk_ex* chunks,
    global partition_segment_result* results,
    global partition_segment* dst_segments,
    global partition_segment_result* dst_results,
    global bitonic_segment* bitonic_segments,
    idx_t bitonic_segments_count,
    idx_t segments_count    
    )
{
    idx_t local_idx = get_local_id(0);
    idx_t group_idx = get_group_id(0);
    partition_segment_chunk_ex chunk_ex = chunks[group_idx];
    partition_segment segment = segments[chunk_ex.segment_idx];
    global partition_segment_result* result = results + chunk_ex.segment_idx;

    local idx_t smaller_than_pivot_global_offset;
    local idx_t greater_than_pivot_global_offset;  
    local idx_t last_group_counter;

    local idx_t chunks_allocate_idx;
    local idx_t bitonic_segments_allocate_idx;
    local idx_t segments_allocate_idx;
    if (local_idx == 0)
    {
        chunks_allocate_idx = 0;
        bitonic_segments_allocate_idx = bitonic_segments_count;
        segments_allocate_idx = 0;
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
        idx_t group_size = get_local_size(0);
        for (idx_t i = local_idx; i < segments_count; i+= group_size)
        {   
            partition_segment segment = segments[i];
            partition_segment_result current_result = results[i];



            

            
            idx_t total_left = current_result.smaller_than_pivot_upper - segment.global_start_idx; 
            idx_t total_right = segment.global_end_idx - current_result.greater_than_pivot_lower; 

            // allocate space for segments
            idx_t segments_to_allocate = (total_left > ALTERNATIVE_SORT_THRESHOLD) + (total_right > ALTERNATIVE_SORT_THRESHOLD);

            idx_t segment_base_idx = 0;
            idx_t chunks_base_idx = 0;
            idx_t chunks_count_left = 0;
            idx_t chunks_count_right = 0;

            if (segments_to_allocate > 0)
            {
                chunks_count_left = total_left / PARTITION_ELEMENTS_PER_WORKGROUP;
                chunks_count_right = total_right / PARTITION_ELEMENTS_PER_WORKGROUP;
                segment_base_idx = atomic_add(&segments_allocate_idx, segments_to_allocate);
                chunks_base_idx = atomic_add(&chunks_allocate_idx, chunks_count_left + chunks_count_right);
            }           
            
            if (total_left > ALTERNATIVE_SORT_THRESHOLD)
            {   
                // left segment calc
                partition_segment left_segment;

                left_segment.global_start_idx = segment.global_start_idx;
                left_segment.global_end_idx = current_result.smaller_than_pivot_upper;
                left_segment.pivot = dst[left_segment.global_start_idx]; // memory consistency?
                dst_segments[segment_base_idx] = left_segment;
                

                for (idx_t i = 0; i < chunks_count_left; ++i)
                {
                    partition_segment_chunk_ex new_chunk;
                    new_chunk.segment_idx = segment_base_idx;
                    new_chunk.chunk.start = left_segment.global_start_idx + i * PARTITION_ELEMENTS_PER_WORKGROUP;
                    new_chunk.chunk.end = ((i+1) == chunks_count_left) ? left_segment.global_end_idx : new_chunk.chunk.start + PARTITION_ELEMENTS_PER_WORKGROUP;
                    chunks[chunks_base_idx + i] = new_chunk;
                    
                }
                chunks_base_idx += chunks_count_left;
                ++segment_base_idx;
            }
            else if (total_left > 0)
            {
                // bitonic sort this segment inplace, or to a different buffer if needed
                idx_t bitonic_idx = atomic_inc(&bitonic_segments_allocate_idx);
                bitonic_segment left_segment;
                left_segment.global_start_idx = segment.global_start_idx;
                left_segment.global_end_idx = current_result.smaller_than_pivot_upper; // +1 ?    
                bitonic_segments[bitonic_idx] = left_segment;
            }

            if (total_right > ALTERNATIVE_SORT_THRESHOLD)
            {               
                // right segment calc
                partition_segment right_segment;
                right_segment.global_start_idx = current_result.greater_than_pivot_lower;
                right_segment.global_end_idx = segment.global_end_idx;
                right_segment.pivot = dst[right_segment.global_start_idx]; // memory consistency?

                dst_segments[segment_base_idx] = right_segment;
                
                for (idx_t i = 0; i < chunks_count_right; ++i)
                {
                    partition_segment_chunk_ex new_chunk;
                    new_chunk.segment_idx = segment_base_idx;
                    new_chunk.chunk.start = right_segment.global_start_idx + i * PARTITION_ELEMENTS_PER_WORKGROUP;
                    new_chunk.chunk.end = ((i+1) == chunks_count_left) ? right_segment.global_end_idx : new_chunk.chunk.start + PARTITION_ELEMENTS_PER_WORKGROUP;
                    chunks[chunks_base_idx + i] = new_chunk;
                    
                }
            }
            else if (total_left > 0)
            {
                // bitonic sort this segment inplace, or to a different buffer if needed
                idx_t bitonic_idx = atomic_inc(&bitonic_segments_allocate_idx);
                bitonic_segment right_segment;
                right_segment.global_start_idx = segment.global_start_idx;
                right_segment.global_end_idx = current_result.smaller_than_pivot_upper; // +1 ?    
                bitonic_segments[bitonic_idx] = right_segment;
            }


            barrier(CLK_LOCAL_MEM_FENCE);

            if (local_idx == 0)
            {
 
                if (chunks_allocate_idx > 0)
                {
                    // dispatch this kernel again using double amount of segments
                    queue_t q = get_default_queue();
                    enqueue_kernel(
                        q,
                        CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
                        ndrange_1D(256 * 5, 256),
                        ^{ 
                        sort2(
                            dst,
                            src,
                            dst_segments,
                            chunks,
                            dst_results,
                            segments,
                            results,                        
                            bitonic_segments,
                            0,
                            0    
                            ); 
                        }
                    
                    ); 
                }

                else if (bitonic_segments_allocate_idx > 0)
                {
                    // dispatch bitonic sort to same destination
                }           
            }            



        }


    }

}
)""