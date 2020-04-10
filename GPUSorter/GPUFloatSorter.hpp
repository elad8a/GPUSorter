#pragma once


namespace HPC
{
    class GPUFloatSorter
    {
    public:
        void Partition();
        void NthElement();
        void Sort();

        void BatchedPartition();
        void BatchedNthElement();
        void BatchedSort();
    private:

    };
}