/*!
 * Copyright (c) 2017 by Contributors
 * \file quantized_convolution.cc
 * \brief
 * \author Ziheng Jiang
*/
#include "./quantized_convolution-inl.h"

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(QuantizedConvolutionParam);

template<>
Operator* CreateOp<cpu>(int dtype,
                        const Context& ctx,
                        const std::vector<TShape>& in_shape,
                        const std::vector<TShape>& out_shape,
                        const QuantizedConvolutionParam& param) {
  LOG(FATAL) << "not implemented yet";
  Operator *op = NULL;
  // MSHADOW_TYPE_SWITCH(dtype, DType, {
  //   op = new QuantizedConvolutionOp<DType>();
  // })
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *QuantizedConvolutionProp::CreateOperatorEx(Context ctx,
    std::vector<TShape> *in_shape, std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, (*in_type)[0], ctx, *in_shape, out_shape, param_);
}

MXNET_REGISTER_OP_PROPERTY(quantized_convolution, QuantizedConvolutionProp)
.add_argument("data", "NDArray-or-Symbol", "Input data.")
.add_argument("filter", "NDArray-or-Symbol", "Weight matrix.")
.add_argument("min_data", "NDArray-or-Symbol", "Minimum value of data.")
.add_argument("max_data", "NDArray-or-Symbol", "Maximum value of data.")
.add_argument("min_filter", "NDArray-or-Symbol", "Minimum value of filter.")
.add_argument("max_filter", "NDArray-or-Symbol", "Maximum value of filter.")
.add_arguments(QuantizedConvolutionParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
