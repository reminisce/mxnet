/*!
 * Copyright (c) 2017 by Contributors
 * \file lowbit_convolution.cc
 * \brief
 * \author Ziheng Jiang
*/
#include "./lowbit_convolution-inl.h"

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(LowbitConvolutionParam);

template<>
Operator* CreateOp<cpu>(int dtype,
                        const Context& ctx,
                        const std::vector<TShape>& in_shape,
                        const std::vector<TShape>& out_shape,
                        const LowbitConvolutionParam& param) {
  LOG(FATAL) << "not implemented yet";
  Operator *op = NULL;
  // MSHADOW_TYPE_SWITCH(dtype, DType, {
  //   op = new LowbitConvolutionOp<DType>();
  // })
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *LowbitConvolutionProp::CreateOperatorEx(Context ctx,
    std::vector<TShape> *in_shape, std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, (*in_type)[0], ctx, *in_shape, out_shape, param_);
}

MXNET_REGISTER_OP_PROPERTY(lowbit_convolution, LowbitConvolutionProp)
.add_argument("data", "NDArray-or-Symbol", "Input data.")
.add_argument("weight", "NDArray-or-Symbol", "Weight matrix.")
.add_argument("bias", "NDArray-or-Symbol", "Bias parameter.")
.add_arguments(LowbitConvolutionParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
