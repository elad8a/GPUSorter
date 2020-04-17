#pragma once
#include "sort_types.hpp"

namespace HPC
{
    class GPUFloatSorter
    {
    public:
        GPUFloatSorter(boost::compute::command_queue& queue, std::size_t maxElements);
        void Sort(boost::compute::vector<int>& src, boost::compute::vector<int>& dst, boost::compute::command_queue& queue);



    private:


        boost::compute::vector<partition_segment> _segments1;
        boost::compute::vector<partition_segment> _segments2;
        boost::compute::vector<partition_segment_result> _results1;
        boost::compute::vector<partition_segment_result> _results2;
        boost::compute::vector<partition_segment_chunk_ex> _chunks;
        boost::compute::vector<bitonic_segment> _bitonicSegments;

        boost::compute::kernel _sortKernel;

        std::vector<partition_segment> _hostSegments;
        std::vector<partition_segment_result> _hostResults;
        std::vector<partition_segment_chunk_ex> _hostChunks;
    
    };
}