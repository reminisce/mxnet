/*!
 * Copyright (c) 2017 by Contributors
 * \file lowbit_relu.cc
 * \brief
 * \author Ziheng Jiang
*/
#include "./lowbit_relu-inl.h"

namespace mxnet {
namespace op {

template<>
Operator *CreateLowbitReluOp<cpu>(int dtype) {
  LOG(FATAL) << "not implemented yet";
  Operator *op = NULL;
  // MSHADOW_TYPE_SWITCH(dtype, DType, {
  //   op = new LowbitReluOp<DType>();
  // })
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *LowbitReluProp::CreateOperatorEx(Context ctx,
                                           std::vector<TShape> *in_shape,
                                           std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateLowbitReluOp, (*in_type)[0]);
}

MXNET_REGISTER_OP_PROPERTY(lowbit_relu, LowbitReluProp)
.describe(R"code(Applies an activation function element-wise to the input.
)code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "Input array to activation function.");

}  // namespace op
}  // namespace mxnet
