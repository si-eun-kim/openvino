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
    // printf("Hello World!\n");
    // printf("Data = %lf, %lf\n", data[0], data[2]);
    // printf("Indices = %f, %lf\n", indices[3], indices[16]);
    // printf("Axis = %d\n", AXIS);
    // printf("Z = %d, Y = %d, X = %d\n", INPUT1_SIZE_Z, INPUT1_SIZE_Y, INPUT1_SIZE_X);
    
    #if INPUT1_DIMS == 4
        // printf("if\n");
        const uint size = INPUT1_BATCH_NUM * INPUT1_FEATURE_NUM * INPUT1_SIZE_X * INPUT1_SIZE_Y;
        size_t data_shape[INPUT1_DIMS] = {INPUT0_BATCH_NUM, INPUT0_FEATURE_NUM, INPUT0_SIZE_Y, INPUT0_SIZE_X};
        size_t indices_shape[INPUT1_DIMS] = {INPUT1_BATCH_NUM, INPUT1_FEATURE_NUM, INPUT1_SIZE_Y, INPUT1_SIZE_X};
    #elif INPUT1_DIMS == 5
        // printf("elif\n");
        const uint size = INPUT1_BATCH_NUM * INPUT1_FEATURE_NUM * INPUT1_SIZE_X * INPUT1_SIZE_Y * INPUT1_SIZE_Z;
        size_t data_shape[INPUT1_DIMS] = {INPUT0_BATCH_NUM, INPUT0_FEATURE_NUM, INPUT0_SIZE_Z, INPUT0_SIZE_Y, INPUT0_SIZE_X};
        size_t indices_shape[INPUT1_DIMS] = {INPUT1_BATCH_NUM, INPUT1_FEATURE_NUM, INPUT1_SIZE_Z, INPUT1_SIZE_Y, INPUT1_SIZE_X};
    #else
        // printf("else\n");
        const uint size = INPUT1_BATCH_NUM * INPUT1_FEATURE_NUM * INPUT1_SIZE_X * INPUT1_SIZE_Y * INPUT1_SIZE_Z * INPUT1_SIZE_W;
        size_t data_shape[INPUT1_DIMS] = {INPUT0_BATCH_NUM, INPUT0_FEATURE_NUM, INPUT0_SIZE_W, INPUT0_SIZE_Z, INPUT0_SIZE_Y, INPUT0_SIZE_X};
        size_t indices_shape[INPUT1_DIMS] = {INPUT1_BATCH_NUM, INPUT1_FEATURE_NUM, INPUT1_SIZE_W, INPUT1_SIZE_Z, INPUT1_SIZE_Y, INPUT1_SIZE_X};
    #endif

    for(size_t i = 0; i < size; ++i)
    {
        int arr[INPUT1_DIMS] = {0};
        int remainder = i;
        int num = 0;
        int mul = 1;

        for(int j = INPUT1_DIMS - 1; j >= 0; --j)
        {
            if (remainder / indices_shape[j] == 0)
            {
                arr[j] = remainder;
                break;
            }
            else
            {
                arr[j] = remainder % indices_shape[j];
                remainder /= indices_shape[j];
            }
        }
        // printf("arr = %d, %d, %d, %d, %d\n", arr[0], arr[1], arr[2], arr[3], arr[4]);

        arr[AXIS] = indices[i];
        // printf("fix = %d, %d, %d, %d, %d\n", arr[0], arr[1], arr[2], arr[3], arr[4]);

        for (int k = INPUT1_DIMS - 1; k >= 0; --k)
        {
            num += arr[k] * mul;
            mul *= data_shape[k];
        }

        // printf("num = %d\n", num);

        output[i] = data[num];
    }

}
