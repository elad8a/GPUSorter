R""(
//#include "partition.cl"

typedef float data_t;
typedef uint idx_t;

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

typedef struct work_descriptor
{
    uint start;
    uint end;
    uint pivot;
    uint direction; // 1 - read from data_1 write to data_2, 0 - the other way. 42 - do not write, nothing to sort
} work_descriptor_t;

//---------------------------------------------------------------------------------------
// kernel implements logic to sort reprocess_work_item records into partition_work and last_stage_work.
// This kernel is launched initially to divide the input sequence into a set of blocks.
// After that it launches gqsort_kernel. The execution alternates between gqsort_kernel
// and relauncher_kernel until all the records are small enough to be processed by the 
// lqsort_kernel. Note two pieces of functionality: sorting reprocess_work into partition_work set and 
// last_stage_work set and subdividing partition_work set into blocks.
//
// data_1 - input array
// data_2 - scratch array of the same size as the input array
// chunks - all the input is split into chunks; each partition_work group is partitioning each 
//          chunk of data around the pivot
// parents - array of parents associated with different chunks
// reprocess_work  - new subarrays generated after partitioning around a pivot
// partition_work    - used for accumulating records that
//           still need processing by recursive_partition
// last_stage_work    - used for accumulating records that 
//           are small enough to be processed by lqsort_kernel
// last_stage_work_count - size of the last_stage_work array
// MAXSEQ  - maximum number of sequences limits
//           the number of chunks passed to recursive_partition for processing
// num_workgroups - parameter used to calculate the number of records in the reprocess_work set.
//---------------------------------------------------------------------------------------
kernel void qsort(
    global data_t* data_1,
    global data_t* data_2,
    global partition_segment_chunk* segments,
    global partition_segment* parents,
    uint   last_stage_work_count, // on first launch it is zero, untouched by other kernels (just pass them through), consider separating
    uint num_workgroups
    )
{
    queue_t q = get_default_queue();


    // stage 1: process reprocess work items
    // this stage processes the results of a previous partition or first invocation
    // it decides which items go to the further partition or to the final stage
    // note that each partition generates 2 reprocess_work items (less & greater than)
    uint partition_work_count = 0;
    global work_descriptor_t* reprocess_work_item = reprocess_work;
    for (uint i = 0; i < 2 * num_workgroups; ++i, ++reprocess_work_item)
    {
        if (reprocess_work_item->direction != EMPTY_RECORD)
        {
            uint reprocess_work_item_elements_count = reprocess_work_item->end - reprocess_work_item->start;
            if (reprocess_work_item_elements_count > QUICKSORT_CHUNK_MIN_SIZE)
            {
                //  work item could refers to a range that can be further partitioned,
                partition_work[partition_work_count] = *reprocess_work_item;
                ++partition_work_count;

            }
            else if (reprocess_work_item_elements_count > 0)
            {
                // if this chunk is compatible for last stage processing, pass it to the last stage work
                last_stage_work[last_stage_work_count] = *reprocess_work_item;
                ++last_stage_work_count; // note this variable is retained between invocations
                
            }
            reprocess_work_item->direction = EMPTY_RECORD;
        }
    }


    // stage 2: prepare chunks for further partition
    // this stage processes the partition_work items that were prepared in the previous stage
    if (partition_work_count > 0) 
    {
        // calculate chunk size, parents and chunks
        uint chunksize = 0;
        global work_descriptor_t* partition_work_item = partition_work;
        for (uint i = 0; i < partition_work_count; ++i, ++partition_work_item)
        {
            uint recsize = (partition_work_item->end - partition_work_item->start) / MAXSEQ;
            if (recsize == 0)
            {
                recsize = 1;
            }
            chunksize += recsize;
        }

        uint parents_size = 0, chunks_count = 0;
        partition_work_item = partition_work;
        for (uint i = 0; i < partition_work_count; ++i, ++partition_work_item)
        {
            uint start = partition_work_item->start;
            uint end = partition_work_item->end;
            uint pivot = partition_work_item->pivot;
            uint direction = partition_work_item->direction;
            uint chunks_count = (end - start + chunksize - 1) / chunksize;
            if (chunks_count == 0)
            {
                chunks_count = 1;
            }
            parent_descriptor_t prnt = { start, end, start, end, chunks_count - 1 };
            parents[i] = prnt;
            ++parents_size;

            for (uint j = 0; j < prnt.chunks_count; ++j) 
            {
                uint bstart = start + chunksize * j;
                chunk_descriptor_t br = { bstart, bstart + chunksize, pivot, direction, parents_size - 1 };
                chunks[chunks_count] = br;
                ++chunks_count;
            }
            chunk_descriptor_t br = { start + chunksize * prnt.chunks_count, end, pivot, direction, parents_size - 1 };
            chunks[chunks_count] = br;
            ++chunks_count;
        }

        // ^{ gqsort_kernel(data_1, data_2, blocks, parents, reprocess_work_item, partition_work, last_stage_work, last_stage_work_count, MAXSEQ, 0); }

        //enqueue_kernel(q, CLK_ENQUEUE_FLAGS_NO_WAIT,
        //    ndrange_8D(GQSORT_LOCAL_WORKGROUP_SIZE * chunks_count, GQSORT_LOCAL_WORKGROUP_SIZE),
        //    nullptr);
    }
    else 
    {

        // all partition work is done, launch final stage
        // ^{ lqsort_kernel(data_1, data_2, last_stage_work); }
        //enqueue_kernel(q, CLK_ENQUEUE_FLAGS_NO_WAIT,
        //    ndrange_8D(LQSORT_LOCAL_WORKGROUP_SIZE * last_stage_work_count, LQSORT_LOCAL_WORKGROUP_SIZE),
        //    nullptr);

    }
}

)""