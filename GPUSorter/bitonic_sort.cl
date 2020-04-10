R""(

typedef float data_t;
typedef uint idx_t;

#define MAX_GROUP_SIZE 256
#define CACHE_SIZE (MAX_GROUP_SIZE * 2)
#define MAX_BITONIC_WORK_SIZE (MAX_GROUP_SIZE * 2)


void bitonic_sort_local_mem(local data_t* cache, idx_t local_idx)
{
    // transform each level to bitonic, assume first level already processed
    for (idx_t level = 2; level < MAX_BITONIC_WORK_SIZE; level <<= 1)
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

            bool flip = !((a < b) ^ dir_flip);
            if (flip)
            {                
               cache[pos] = b;
               cache[pos + merge_level] = a;
            }
            barrier(CLK_LOCAL_MEM_FENCE);      
         }
    }
}



__kernel void bitonic_sort(__global data_t* input, idx_t input_size)
{    
    __local data_t cache[CACHE_SIZE];
    
    idx_t local_idx = get_local_id(0);
    idx_t access_base = local_idx * 2;
  
    data_t a = access_base < input_size ? input[access_base] : FLT_MAX;
    data_t b = (access_base + 1) < input_size ? input[access_base + 1] : FLT_MAX;

    bool flip = !(a < b) ^ (local_idx & 1);
    if (flip)
    {                
       cache[access_base] = b;
       cache[access_base + 1] = a;
    }
    else
    {       
        cache[access_base] = a;
        cache[access_base + 1] = b;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    bitonic_sort_local_mem(cache, local_idx);

    if (access_base < input_size)
    {    
        input[access_base] = cache[access_base];
        if ((access_base + 1) < input_size)
        {     
            input[access_base + 1] = cache[access_base + 1];
        }  
    }
}




)""