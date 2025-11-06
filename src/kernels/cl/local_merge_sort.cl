#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#define min(a,b) ((a)<(b)?(a):(b))

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void local_merge_sort(
    __global const uint* input_data,
    __global       uint* output_data)
{
    __local uint buffer_a[TILE_SIZE];
    __local uint buffer_b[TILE_SIZE];

    const uint tile_start = get_group_id(0) * TILE_SIZE;
    const uint local_id = get_local_id(0);

    for (int i = 0; i < N_ELEMENTS_PER_THREAD; ++i) {
        uint global_read_index = tile_start + i * GROUP_SIZE + local_id;
        uint local_write_index = i * GROUP_SIZE + local_id;
        buffer_a[local_write_index] = input_data[global_read_index];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    __local uint* src = buffer_a;
    __local uint* dst = buffer_b;

    for (int merge_width = 1; merge_width < TILE_SIZE; merge_width <<= 1) {
        for (int i = 0; i < N_ELEMENTS_PER_THREAD; ++i) {
            uint index = i * GROUP_SIZE + local_id;
            uint value = src[index];

            uint merge_start_index = index & (~(merge_width * 2 - 1));
            uint mid = min(merge_start_index + merge_width, TILE_SIZE);
            bool is_left = index < mid;

            int bs_start = is_left ? mid : merge_start_index;
            int bs_end = min(bs_start + merge_width, TILE_SIZE);
            
            int l = bs_start - 1, r = bs_end;
            while (l + 1 < r) {
                int m = (l + r) / 2;
                if (src[m] < value || (src[m] == value && is_left)) {
                    l = m;
                } else {
                    r = m;
                }
            }

            dst[merge_start_index + (index - merge_start_index - (is_left ? 0 : merge_width)) + (r - bs_start)] = value;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        __local uint* tmp = src;
        src = dst;
        dst = tmp;
    }

    for (int i = 0; i < N_ELEMENTS_PER_THREAD; ++i) {
        uint local_read_index = i * GROUP_SIZE + local_id;
        uint global_write_index = tile_start + i * GROUP_SIZE + local_id;
        output_data[global_write_index] = src[local_read_index];
    }
}