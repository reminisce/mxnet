/*!
 * Copyright (c) 2017 by Contributors
 * \file lowbit_max_pool.cc
 * \brief
 * \author Ziheng Jiang
*/
#include "./lowbit_max_pool-inl.h"

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(LowbitMaxPoolParam param, int dtype) {
  Operator *op = NULL;
  LOG(FATAL) << "not implemented";
  // MSHADOW_TYPE_SWITCH(dtype, DType, {
  //   op = new LowbitMaxPoolOp<cpu, DType>(param);
  // });
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator* LowbitMaxPoolProp::CreateOperatorEx(Context ctx,
  std::vector<TShape> *in_shape, std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(LowbitMaxPoolParam);

MXNET_REGISTER_OP_PROPERTY(lowbit_max_pool, LowbitMaxPoolProp)
.describe(R"code(Performs pooling on the input.

The shapes for 1-D pooling are

- **data**: *(batch_size, channel, width)*,
- **out**: *(batch_size, num_filter, out_width)*.

The shapes for 2-D pooling are

- **data**: *(batch_size, channel, height, width)*
- **out**: *(batch_size, num_filter, out_height, out_width)*, with::

    out_height = f(height, kernel[0], pad[0], stride[0])
    out_width = f(width, kernel[1], pad[1], stride[1])

The defintion of *f* depends on ``pooling_convention``, which has two options:

- **valid** (default)::

    f(x, k, p, s) = floor(x+2*p-k)/s+1

- **full**, which is compatible with Caffe::

    f(x, k, p, s) = ceil(x+2*p-k)/s+1

But ``global_pool`` is set to be true, then do a global pooling, namely reset
``kernel=(height, width)``.

For 3-D pooling, an additional *depth* dimension is added before
*height*. Namely the input data will have shape *(batch_size, channel, depth,
height, width)*.

)code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "Input data to the pooling operator.")
.add_arguments(LowbitMaxPoolParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
