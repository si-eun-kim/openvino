// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/fetch_data.cl"

KERNEL(gather_elements_ref)(const __global INPUT0_TYPE* data,
                   const __global INPUT1_TYPE* indices,
                   __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
                   , FUSED_OPS_DECLS
#endif
)
{
    const uint dim0 = get_global_id(0);
    const uint dim1 = get_global_id(1);
    const uint dim2 = get_global_id(2);
    
    #if INPUT1_DIMS == 4
        const uint x = dim0;
        const uint y = dim1;
    #elif INPUT1_DIMS == 5
        const uint x = dim0;
        const uint y = dim1 % OUTPUT_SIZE_Y;
        const uint z = dim1 / OUTPUT_SIZE_Y;
    #else
        const uint x = dim0 % OUTPUT_SIZE_X;
        const uint y = dim0 / OUTPUT_SIZE_X;
        const uint z = dim1 % OUTPUT_SIZE_Z;
        const uint w = dim1 / OUTPUT_SIZE_Z;
    #endif
        const uint f = dim2 % OUTPUT_FEATURE_NUM;
        const uint b = dim2 / OUTPUT_FEATURE_NUM;

    #if INPUT1_DIMS == 4
        int arr[INPUT1_DIMS] = {b, f, y, x};
        size_t data_shape[INPUT1_DIMS] = {INPUT0_BATCH_NUM, INPUT0_FEATURE_NUM, INPUT0_SIZE_Y, INPUT0_SIZE_X};
        size_t indices_shape[INPUT1_DIMS] = {INPUT1_BATCH_NUM, INPUT1_FEATURE_NUM, INPUT1_SIZE_Y, INPUT1_SIZE_X};
    #elif INPUT1_DIMS == 5
        int arr[INPUT1_DIMS] = {b, f, z, y, x};
        size_t data_shape[INPUT1_DIMS] = {INPUT0_BATCH_NUM, INPUT0_FEATURE_NUM, INPUT0_SIZE_Z, INPUT0_SIZE_Y, INPUT0_SIZE_X};
        size_t indices_shape[INPUT1_DIMS] = {INPUT1_BATCH_NUM, INPUT1_FEATURE_NUM, INPUT1_SIZE_Z, INPUT1_SIZE_Y, INPUT1_SIZE_X};
    #else
        int arr[INPUT1_DIMS] = {b, f, w, z, y, x};
        size_t data_shape[INPUT1_DIMS] = {INPUT0_BATCH_NUM, INPUT0_FEATURE_NUM, INPUT0_SIZE_W, INPUT0_SIZE_Z, INPUT0_SIZE_Y, INPUT0_SIZE_X};
        size_t indices_shape[INPUT1_DIMS] = {INPUT1_BATCH_NUM, INPUT1_FEATURE_NUM, INPUT1_SIZE_W, INPUT1_SIZE_Z, INPUT1_SIZE_Y, INPUT1_SIZE_X};
    #endif

    int out_idx = 0, mul = 1, data_idx = 0;

    for(int i = INPUT1_DIMS - 1; i >= 0; --i)
    {
        out_idx += arr[i] * mul;
        mul *= indices_shape[i];
    }
    
    arr[AXIS] = indices[out_idx];

    mul = 1;
    for(int j = INPUT1_DIMS - 1; j >= 0; --j)
    {
        data_idx += arr[j] * mul;
        mul *= data_shape[j];
    }

    //output[out_idx] = data[data_idx];
    INPUT0_TYPE val = data[data_idx];

    #if HAS_FUSED_OPS
        FUSED_OPS;
        output[out_idx] = TO_OUTPUT_TYPE(FUSED_OPS_RESULT);
    #else
        output[out_idx] = ACTIVATION(val, ACTIVATION_PARAMS);
    #endif
}