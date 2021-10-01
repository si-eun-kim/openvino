// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief
/// @details
struct gather_elements : public primitive_base<gather_elements> {
    CLDNN_DECLARE_PRIMITIVE(gather_elements)

    /// @brief Constructs gather_elements primitive.
    /// @param id This primitive id.
    /// @param data Input data primitive id.
    /// @param indices Input indexes primitive id.
    /// @param indices_rank Rank of indices.
    /// @param batch_dims batch_dims as an attribute of GatherElements. Optional.
    gather_elements(const primitive_id& id,
              const primitive_id& data,
              const primitive_id& indices,
              const uint8_t indices_rank,
              const uint8_t batch_dims = 0,
              const primitive_id& ext_prim_id = "",
              const padding& output_padding = padding())
        : primitive_base(id, {data, indices}, ext_prim_id, output_padding), indices_rank(indices_rank), batch_dims(batch_dims) {}

    /// @brief GatherElements indices_rank
    uint8_t indices_rank;

    /// @brief GatherElements batch_dims
    uint8_t batch_dims;
};
/// @}
/// @}
/// @}
}  // namespace cldnn
