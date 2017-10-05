/*!
 *  Copyright (c) 2017 by Contributors
 * \file quantize.cc
 * \brief
 */
#include "./quantize_down_and_shrink_range-inl.h"

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(QuantizeDownAndShrinkRangeParam);

NNVM_REGISTER_OP(quantize_down_and_shrink_range)
.set_attr_parser(ParamParser<QuantizeDownAndShrinkRangeParam>)
.set_num_inputs(3)
.set_num_outputs(3)
.set_attr<nnvm::FInferShape>("FInferShape", QuantizeDownAndShrinkRangeShape)
.set_attr<nnvm::FInferType>("FInferType", QuantizeDownAndShrinkRangeType)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>(3, ResourceRequest::kTempSpace);
  })
.add_argument("data", "NDArray-or-Symbol", "A ndarray/symbol of type `int32`")
.add_argument("min_range", "NDArray-or-Symbol", "The original minimum scalar value "
  "in the form of float32 possibly produced for the input")
.add_argument("max_range", "NDArray-or-Symbol", "The original maximum scalar value "
  "in the form of float32 possibly produced for the input")
.add_arguments(QuantizeDownAndShrinkRangeParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
