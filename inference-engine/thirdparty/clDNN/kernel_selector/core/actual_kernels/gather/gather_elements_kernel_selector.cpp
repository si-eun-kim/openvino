// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_elements_kernel_selector.h"
#include "gather_elements_kernel_ref.h"

namespace kernel_selector {

gather_elements_kernel_selector::gather_elements_kernel_selector() { Attach<GatherElementsKernelRef>(); }

KernelsData gather_elements_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::GATHER_ELEMENTS);
}
}  // namespace kernel_selector
