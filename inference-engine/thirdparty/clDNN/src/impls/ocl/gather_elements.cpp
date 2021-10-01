// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_elements_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "gather/gather_elements_kernel_selector.h"
#include "gather/gather_elements_kernel_ref.h"

using namespace cldnn;

namespace cldnn {
namespace ocl {

struct gather_elements_impl : typed_primitive_impl_ocl<gather_elements> {
    using parent = typed_primitive_impl_ocl<gather_elements>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<gather_elements_impl>(*this);
    }

    static primitive_impl* create(const gather_elements_node& arg) {
        auto gather_elements_params = get_default_params<kernel_selector::gather_elements_params>(arg);
        auto gather_elements_optional_params =
            get_default_optional_params<kernel_selector::gather_elements_optional_params>(arg.get_program());

        gather_elements_params.indices_rank = arg.get_primitive()->indices_rank;
        gather_elements_params.batch_dims = arg.get_primitive()->batch_dims;

        gather_elements_params.inputs.push_back(convert_data_tensor(arg.input(1).get_output_layout()));

        auto& kernel_selector = kernel_selector::gather_elements_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(gather_elements_params, gather_elements_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto gather_elements = new gather_elements_impl(arg, best_kernels[0]);

        return gather_elements;
    }
};

namespace detail {

attach_gather_elements_impl::attach_gather_elements_impl() {
    implementation_map<gather_elements>::add(impl_types::ocl, gather_elements_impl::create, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::i32, format::bfyx),
        std::make_tuple(data_types::f32, format::bfzyx),
        std::make_tuple(data_types::f16, format::bfzyx),
        std::make_tuple(data_types::i32, format::bfzyx),
        std::make_tuple(data_types::f32, format::bfwzyx),
        std::make_tuple(data_types::f16, format::bfwzyx),
        std::make_tuple(data_types::i32, format::bfwzyx),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
