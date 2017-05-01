/*!
 * Copyright (c) 2017 by Contributors
 * \file lowbit_fully_connected.cc
 * \brief
 * \author Ziheng Jiang
*/
#include "./lowbit_fully_connected-inl.h"

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(LowbitFullyConnectedParam);

template<>
Operator* CreateOp<cpu>(int dtype,
                        const Context& ctx,
                        const std::vector<TShape>& in_shape,
                        const std::vector<TShape>& out_shape,
                        const LowbitFullyConnectedParam& param) {
  LOG(FATAL) << "not implemented yet";
  Operator *op = NULL;
  // MSHADOW_TYPE_SWITCH(dtype, DType, {
  //   op = new LowbitFullyConnectedOp<DType>();
  // })
  return op;
}

Operator *LowbitFullyConnectedProp::CreateOperatorEx(Context ctx,
    std::vector<TShape> *in_shape, std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, (*in_type)[0], ctx, *in_shape, out_shape, param_);
}

MXNET_REGISTER_OP_PROPERTY(lowbit_fully_connected, LowbitFullyConnectedProp)
.add_argument("data", "NDArray-or-Symbol", "Input data.")
.add_argument("weight", "NDArray-or-Symbol", "Weight matrix.")
.add_arguments(LowbitFullyConnectedParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
