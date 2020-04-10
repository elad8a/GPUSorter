R""(

typedef float data_t;
typedef uint idx_t;

#define MAX_GROUP_SIZE 256
#define CACHE_SIZE (MAX_GROUP_SIZE * 2)
#define MAX_BITONIC_WORK_SIZE (MAX_GROUP_SIZE * 2)

__kernel void bitonic_sort(__global data_t* input)
{    
    __local data_t cache[CACHE_SIZE];
    
    idx_t local_idx = get_local_id(0);

    cache[local_idx] = input[local_idx];
    cache[local_idx + MAX_GROUP_SIZE] = input[local_idx + MAX_GROUP_SIZE];
    barrier(CLK_LOCAL_MEM_FENCE);
    

    //
    // stage 1: transform to bitonic
    //
    // transform each level to bitonic
    for (idx_t level = 1; level < MAX_BITONIC_WORK_SIZE; level <<= 1)
    {
        bool dir_flip = local_idx & level;

        // bitonic merge each level
        for (idx_t merge_level = level; merge_level > 0; merge_level >>= 1)
        {
            idx_t merge_group = local_idx / merge_level;
            idx_t idx_within_group = local_idx % merge_level;
            idx_t pos = merge_group * merge_level * 2 + idx_within_group;
            data_t a = cache[pos];
            data_t b = cache[pos + merge_level];

            bool cond = (a < b) ^ dir_flip;
            if (!cond)
            {                
               cache[pos] = b;
               cache[pos + merge_level] = a;
            }
            barrier(CLK_LOCAL_MEM_FENCE);      
         }
    }

    input[local_idx] = cache[local_idx];
    input[local_idx + MAX_GROUP_SIZE] = cache[local_idx + MAX_GROUP_SIZE];
}

)""
