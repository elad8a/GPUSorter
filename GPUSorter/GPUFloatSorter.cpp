#include "pch.hpp"
#include "GPUFloatSorter.hpp"

inline const std::string_view PARTITION_KERNEL_STRING =
#include "../GPUSorter/partition.cl"
"";
inline const std::string_view SORT_KERNEL_STRING =
#include "../GPUSorter/sort.cl"
"";

inline const std::string_view BITONIC_SORT_KERNEL_STRING =
#include "../GPUSorter/bitonic_sort.cl"
"";

inline constexpr unsigned DEFAULT_GROUP_SIZE = 256;
inline constexpr unsigned DEFAULT_MAX_ELEMENTS_PER_CHUNK = DEFAULT_GROUP_SIZE * 4;

namespace bc = boost::compute;



namespace HPC
{
    GPUFloatSorter::GPUFloatSorter(boost::compute::command_queue& queue, std::size_t maxElements)
        :
        _segments1(maxElements / DEFAULT_GROUP_SIZE, queue.get_context()),
        _segments2(maxElements / DEFAULT_GROUP_SIZE, queue.get_context()),
        _results1(maxElements / DEFAULT_GROUP_SIZE, queue.get_context()),
        _results2(maxElements / DEFAULT_GROUP_SIZE, queue.get_context()),
        _chunks(maxElements / DEFAULT_GROUP_SIZE, queue.get_context()),
        _bitonicSegments(maxElements / DEFAULT_GROUP_SIZE, queue.get_context()),
        _hostSegments(maxElements / DEFAULT_GROUP_SIZE),
        _hostResults(maxElements / DEFAULT_GROUP_SIZE),
        _hostChunks(maxElements / DEFAULT_GROUP_SIZE)
    {

        const char* strings[3]
        {
            PARTITION_KERNEL_STRING.data(),
            SORT_KERNEL_STRING.data(),
            BITONIC_SORT_KERNEL_STRING.data()

        };

        std::size_t lengths[3]
        {
            PARTITION_KERNEL_STRING.size(),
            SORT_KERNEL_STRING.size(),
            BITONIC_SORT_KERNEL_STRING.size()

        }; 


        auto options = boost::format(
            "-w -cl-std=CL2.0 "
            "-D TARGET_TYPE=%1% "
            "-D MAX_GROUP_SIZE=%2% "
            "-D PARTITION_ELEMENTS_PER_WORKGROUP=%4% ") 
            % ("int")
            % (DEFAULT_GROUP_SIZE)
            % (DEFAULT_GROUP_SIZE * 2)
            % (DEFAULT_MAX_ELEMENTS_PER_CHUNK)
            ;
        cl_int err;
        auto prog = clCreateProgramWithSource(queue.get_context(), 2, strings, lengths, &err);
        assert(err == CL_SUCCESS);
        auto program = bc::program(prog, false);
        program.build(options.str());

        _sortKernel = program.create_kernel("sort");
    }
    void GPUFloatSorter::Sort(boost::compute::vector<int>& src, boost::compute::vector<int>& dst, boost::compute::command_queue& queue)
    {
        assert(src.size() == dst.size());
        _hostSegments.resize(1);

        _hostSegments[0].global_start_idx = 0;
        _hostSegments[0].global_end_idx = (idx_t)src.size();
        _hostSegments[0].pivot = src[0];

        auto chunk_count = (src.size() + DEFAULT_MAX_ELEMENTS_PER_CHUNK - 1) / DEFAULT_MAX_ELEMENTS_PER_CHUNK;
        _hostChunks.resize(chunk_count);

        auto lastChunkIdx = chunk_count - 1;
        for (auto i = 0; i <= lastChunkIdx; ++i)
        {
            auto& chunk = _hostChunks[i];

            chunk.segment_idx = 0;
            chunk.chunk.start = DEFAULT_MAX_ELEMENTS_PER_CHUNK * i;
            chunk.chunk.end = (i == lastChunkIdx) ? _hostSegments[0].global_end_idx :  DEFAULT_MAX_ELEMENTS_PER_CHUNK * (i + 1);
        }

      
        bc::copy_async(_hostChunks.begin(), _hostChunks.end(), _chunks.begin(), queue);
        bc::copy_async(_hostSegments.begin(), _hostSegments.end(), _segments1.begin(), queue);

        auto global_size = chunk_count * DEFAULT_GROUP_SIZE;
        _sortKernel.set_args(
            src,
            dst,
            _segments1,
            _chunks,
            _results1,
            _segments2,
            _results2,
            _bitonicSegments,
            0,
            1
            );

        queue.enqueue_1d_range_kernel(
            _sortKernel,
            0,
            global_size,
            DEFAULT_GROUP_SIZE
            );
    }
}