/*!
 * Copyright (c) 2017 by Contributors
 * \file quantized_bias_add.cc
 * \brief
 * \author Ziheng Jiang
*/
#include "./quantized_bias_add-inl.h"

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(QuantizedBiasAddParam);

template<>
Operator* CreateOp<cpu>(int dtype,
                        const Context& ctx,
                        const std::vector<TShape>& in_shape,
                        const std::vector<TShape>& out_shape,
                        const QuantizedBiasAddParam& param) {
  LOG(FATAL) << "not implemented yet";
  Operator *op = NULL;
  // MSHADOW_TYPE_SWITCH(dtype, DType, {
  //   op = new QuantizedBiasAddOp<DType>();
  // })
  return op;
}

Operator *QuantizedBiasAddProp::CreateOperatorEx(Context ctx,
    std::vector<TShape> *in_shape, std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, (*in_type)[0], ctx, *in_shape, out_shape, param_);
}

MXNET_REGISTER_OP_PROPERTY(quantized_bias_add, QuantizedBiasAddProp)
.add_argument("data", "NDArray-or-Symbol", "matrix a")
.add_argument("bias", "NDArray-or-Symbol", "matrix b")
.add_argument("min_data", "NDArray-or-Symbol", "minimum value of matrix a")
.add_argument("max_data", "NDArray-or-Symbol", "maximum value of matrix a")
.add_argument("min_bias", "NDArray-or-Symbol", "minimum value of matrix b")
.add_argument("max_bias", "NDArray-or-Symbol", "maximum value of matrix b")
.add_arguments(QuantizedBiasAddParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
