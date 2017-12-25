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

template<typename xpu, typename DType>
size_t ConfigReduce(mshadow::Stream<xpu>* s,
                    const TShape& data_shape,
                    const TShape& out_shape,
                    TShape* src_shape,
                    TShape* dst_shape) {
  //TShape src_shape, dst_shape;
  BroadcastReduceShapeCompact(data_shape, out_shape, src_shape, dst_shape);
  constexpr int NDim = 2;
  CHECK_EQ(src_shape->ndim(), NDim);
  CHECK_EQ(dst_shape->ndim(), NDim);

  return broadcast::ReduceWorkspaceSize<NDim, DType>(s, *dst_shape, kWriteTo, *src_shape);
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
  const QuantizeDownAndShrinkRangeParam& param =
    nnvm::get<QuantizeDownAndShrinkRangeParam>(attrs.parsed);

  if (param.min_fval.has_value() && param.max_fval.has_value()) {  // model is calibrated
    Kernel<RequantizeManyInNewRangeStruct, gpu>::Launch(s, inputs[0].Size(),
        outputs[0].dptr<DstDType>(), outputs[1].dptr<float>(), outputs[2].dptr<float>(),
        inputs[0].dptr<SrcDType>(), inputs[1].dptr<float>(), inputs[2].dptr<float>(),
        param.min_fval.value(), param.max_fval.value());
  } else { // model is not calibrated
    TShape src_shape, dst_shape;
    const size_t actual_float_size = sizeof(float);
    const size_t actual_quantized_size = sizeof(SrcDType);
    const size_t temp_reduce_size = ConfigReduce<gpu, SrcDType>(s, inputs[0].shape_, TShape({1}), &src_shape, &dst_shape);
    Tensor<gpu, 1, char> temp_space =
      ctx.requested[0].get_space_typed<gpu, 1, char>(Shape1(2*actual_float_size+2*actual_quantized_size+temp_reduce_size), s);
    Tensor<gpu, 1, float> actual_min_float(reinterpret_cast<float*>(temp_space.dptr_), Shape1(1), s);
    Tensor<gpu, 1, float> actual_max_float(reinterpret_cast<float*>(temp_space.dptr_) + 1, Shape1(1), s);

    const int dev_id = ctx.run_ctx.ctx.dev_id;
    TBlob actual_min_quantized(reinterpret_cast<SrcDType*>(temp_space.dptr_ + 8), Shape1(1), gpu::kDevMask, dev_id);
    TBlob actual_max_quantized(reinterpret_cast<SrcDType*>(temp_space.dptr_ + 8) + 1, Shape1(1), gpu::kDevMask, dev_id);
    Tensor<gpu, 1, char> workspace(temp_space.dptr_+2*actual_float_size+2*actual_quantized_size, Shape1(temp_reduce_size), s);
    //Reduce<red::minimum, SrcDType>(ctx, actual_min_quantized, inputs[0], req_cnt++);
    broadcast::Reduce<red::minimum, 2, SrcDType, mshadow::op::identity>(
      s, actual_min_quantized.reshape(dst_shape), kWriteTo, workspace, inputs[0].reshape(src_shape));
    Kernel<QuantizedToFloatStruct, gpu>::Launch(s, 1,
        actual_min_float.dptr_, actual_min_quantized.dptr<SrcDType>(),
        inputs[1].dptr<float>(), inputs[2].dptr<float>());

    //Reduce<red::maximum, SrcDType>(ctx, actual_max_quantized, inputs[0], req_cnt++);
    broadcast::Reduce<red::maximum, 2, SrcDType, mshadow::op::identity>(
      s, actual_max_quantized.reshape(dst_shape), kWriteTo, workspace, inputs[0].reshape(src_shape));
    Kernel<QuantizedToFloatStruct, gpu>::Launch(s, 1,
        actual_max_float.dptr_, actual_max_quantized.dptr<SrcDType>(),
        inputs[1].dptr<float>(), inputs[2].dptr<float>());

    Kernel<RequantizeManyInNewRangeStruct, gpu>::Launch(s, inputs[0].Size(),
        outputs[0].dptr<DstDType>(), outputs[1].dptr<float>(), outputs[2].dptr<float>(),
        inputs[0].dptr<SrcDType>(), inputs[1].dptr<float>(), inputs[2].dptr<float>(),
        actual_min_float.dptr_, actual_max_float.dptr_);
  }
}

NNVM_REGISTER_OP(quantize_down_and_shrink_range)
.set_attr<FCompute>("FCompute<gpu>", QuantizeDownAndShrinkRangeComputeGPU);

}  // namespace op
}  // namespace mxnet
