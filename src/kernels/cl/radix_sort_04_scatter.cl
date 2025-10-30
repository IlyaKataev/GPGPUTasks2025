#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* input_data,
    __global const uint* scanned_histograms,
    __global       uint* output_data,
    uint n,
    uint offset)
{
    __local uint digit_counters[NUM_DIGITS];

    const uint local_id = get_local_id(0);
    const uint global_id = get_global_id(0);
    const uint group_id = get_group_id(0);
    const uint num_groups = get_num_groups(0);

    if (local_id < NUM_DIGITS) {
        digit_counters[local_id] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint value = 0;
    uint digit = 0;
    if (global_id < n) {
        value = input_data[global_id];
        digit = (input_data[global_id] >> offset) & RADIX_MASK;
    }
    
    uint local_offset = 0;
    if (global_id < n) {
        local_offset = atomic_add(&digit_counters[digit], 1);
    }
    
    uint global_offset = 0;
    uint hist_index = digit * num_groups + group_id;
    if (hist_index > 0) {
        global_offset = scanned_histograms[hist_index - 1];
    }
    
    if (global_id < n) {
        output_data[global_offset + local_offset] = value;
    }
}