/*!
 * Copyright (c) 2017 by Contributors
 * \file quantized_matmul.cc
 * \brief
 * \author Ziheng Jiang
*/
#include "./quantized_matmul-inl.h"

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(QuantizedMatmulParam);

template<>
Operator* CreateOp<cpu>(int dtype,
                        const Context& ctx,
                        const std::vector<TShape>& in_shape,
                        const std::vector<TShape>& out_shape,
                        const QuantizedMatmulParam& param) {
  LOG(FATAL) << "not implemented yet";
  Operator *op = NULL;
  // MSHADOW_TYPE_SWITCH(dtype, DType, {
  //   op = new QuantizedMatmulOp<DType>();
  // })
  return op;
}

Operator *QuantizedMatmulProp::CreateOperatorEx(Context ctx,
    std::vector<TShape> *in_shape, std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, (*in_type)[0], ctx, *in_shape, out_shape, param_);
}

MXNET_REGISTER_OP_PROPERTY(quantized_matmul, QuantizedMatmulProp)
.add_argument("a", "NDArray-or-Symbol", "matrix a")
.add_argument("b", "NDArray-or-Symbol", "matrix b")
.add_argument("min_a", "NDArray-or-Symbol", "minimum value of matrix a")
.add_argument("max_a", "NDArray-or-Symbol", "maximum value of matrix a")
.add_argument("min_b", "NDArray-or-Symbol", "minimum value of matrix b")
.add_argument("max_b", "NDArray-or-Symbol", "maximum value of matrix b")
.add_arguments(QuantizedMatmulParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
