#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   int  sorted_k,
                   int  n)
{
    const uint i = get_global_id(0);

    uint value = input_data[i];

    uint merge_start_index = i & (~(sorted_k * 2 - 1));
    uint mid = merge_start_index + sorted_k;
    
    if (mid >= n) {
        output_data[i] = value;
        return;
    }

    bool is_left = i < mid;

    int bs_start = (is_left ? mid : merge_start_index);
    int bs_end = min(bs_start + sorted_k, n);

    int l = bs_start - 1, r = bs_end;

    while (l + 1 < r) {
        int m = (l + r) / 2;
        if (input_data[m] < value || (input_data[m] == value && is_left)) {
            l = m;
        } else {
            r = m;
        }
    }

    output_data[merge_start_index + (i - (is_left ? merge_start_index : mid)) + (r - bs_start)] = value;
}
