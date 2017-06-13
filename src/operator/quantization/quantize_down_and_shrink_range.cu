/*!
 *  Copyright (c) 2017 by Contributors
 * \file quantize.cu
 * \brief
 */
#include <limits>
#include "./quantize_down_and_shrink_range-inl.h"
#include "./quantization_utils.h"
#include "../tensor/broadcast_reduce_op.h"

namespace mxnet {
namespace op {

template<typename Reducer, typename xpu, typename SrcExp, typename DType>
static void ReduceToAssign(mshadow::Tensor<xpu, 2, DType> out, const OpReqType req, const SrcExp &src) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Shape<2> src_shape = ShapeCheck<2, SrcExp>::Check(src);
  if (src_shape == out.shape_) {
    ASSIGN_DISPATCH(out, req, F<mshadow_op::identity>(src));
  } else if (src_shape[0] == out.shape_[0]) {
    ASSIGN_DISPATCH(out.FlatTo1D(), req, (reduce_except_dim<0, Reducer>(src)));
  } else if (src_shape[1] == out.shape_[1]) {
    ASSIGN_DISPATCH(out.FlatTo1D(), req, (reduce_except_dim<1, Reducer>(src)));
  } else {
    ASSIGN_DISPATCH(out.FlatTo1D(), req,
      (reduce_except_dim<1, Reducer>(reshape(src,
      Shape2(src_shape.Size(), 1)))));
  }
}


template<typename Reducer, typename DType, typename T1, typename T2, typename TStream>
static void Reduce(T1 out, T2 data, TStream *s) {
  TShape src_shape, dst_shape;
  BroadcastReduceShapeCompact(data.shape_, out.shape_, &src_shape, &dst_shape);

  CHECK_EQ(dst_shape.ndim(), 2);
  CHECK_EQ(src_shape.ndim(), 2);
  mshadow::Tensor<gpu, 2, DType>  tout(out.dptr_,  dst_shape.get<2>(), s);
  mshadow::Tensor<gpu, 2, DType> tdata(data.dptr_, src_shape.get<2>(), s);
  ReduceToAssign<Reducer>(tout, kWriteTo, tdata);
}

void QuantizeDownAndShrinkRangeComputeGPU(
    const nnvm::NodeAttrs& attrs,
    const OpContext& ctx,
    const std::vector<TBlob>& inputs,
    const std::vector<OpReqType>& req,
    const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  typedef int32_t SrcDType;
  typedef int8_t  DstDType;
  Stream<gpu> *s = ctx.get_stream<gpu>();

  size_t space_size = 2 * sizeof(float) + 2 * sizeof(SrcDType);
  Tensor<gpu, 1, char> space =
    ctx.requested[0].get_space_typed<gpu, 1, char>(Shape1(space_size), s);

  Tensor<gpu, 1, SrcDType> actual_min_quantized(
    reinterpret_cast<SrcDType*>(space.dptr_ + 8), Shape1(1), s);
  Tensor<gpu, 1, SrcDType> actual_max_quantized(
    reinterpret_cast<SrcDType*>(space.dptr_ + 8) + 1, Shape1(1), s);

  Tensor<gpu, 2, SrcDType> data = inputs[0].FlatTo2D<gpu, SrcDType>(s);
  Reduce<red::minimum, SrcDType>(actual_min_quantized, data, s);
  Reduce<red::maximum, SrcDType>(actual_max_quantized, data, s);

  Tensor<gpu, 1, float> actual_min_float(
    reinterpret_cast<float*>(space.dptr_), Shape1(1), s);
  Tensor<gpu, 1, float> actual_max_float(
    reinterpret_cast<float*>(space.dptr_) + 1, Shape1(1), s);

  Kernel<QuantizedToFloatStruct, gpu>::Launch(s, 1,
      actual_min_float.dptr_, actual_min_quantized.dptr_,
      inputs[1].dptr<float>(), inputs[2].dptr<float>());
  Kernel<QuantizedToFloatStruct, gpu>::Launch(s, 1,
      actual_max_float.dptr_, actual_max_quantized.dptr_,
      inputs[1].dptr<float>(), inputs[2].dptr<float>());

  Kernel<RequantizeManyInNewRangeStruct, gpu>::Launch(s, inputs[0].Size(),
      outputs[0].dptr<DstDType>(), outputs[1].dptr<float>(), outputs[2].dptr<float>(),
      inputs[0].dptr<SrcDType>(), inputs[1].dptr<float>(), inputs[2].dptr<float>(),
      actual_min_float.dptr_, actual_max_float.dptr_);
}

NNVM_REGISTER_OP(quantize_down_and_shrink_range)
.set_attr<FCompute>("FCompute<gpu>", QuantizeDownAndShrinkRangeComputeGPU);

}  // namespace op
}  // namespace mxnet
