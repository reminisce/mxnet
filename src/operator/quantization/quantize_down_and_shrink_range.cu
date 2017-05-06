/*!
 *  Copyright (c) 2017 by Contributors
 * \file quantize.cu
 * \brief
 */
#include <limits>
#include "./quantize_down_and_shrink_range-inl.h"
#include "./quantization_utils.h"

namespace mxnet {
namespace op {

// TODO(ziheng) move to GPU
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

  size_t size = inputs[0].shape_.Size();
  SrcDType *data = (SrcDType *)malloc(size * sizeof(SrcDType));
  float input_min_float;
  float input_max_float;
  cudaMemcpy(data, inputs[0].dptr_, size * sizeof(SrcDType), cudaMemcpyDeviceToHost);
  cudaMemcpy(&input_min_float, inputs[1].dptr_, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&input_max_float, inputs[2].dptr_, sizeof(float), cudaMemcpyDeviceToHost);
  DstDType *out = (DstDType *)malloc(size * sizeof(DstDType));

  float actual_min_quantized = std::numeric_limits<int32_t>::max();
  float actual_max_quantized = std::numeric_limits<int32_t>::min();
  for (size_t i = 0; i < size; ++i) {
    const float value = static_cast<float>(data[i]);
    actual_min_quantized = std::min(actual_min_quantized, value);
    actual_max_quantized = std::max(actual_max_quantized, value);
  }

  // (TODO) We want to make sure that the minimum is no larger
  // than zero, so that the convolution operation can run efficiently.
  const float actual_min_float = QuantizedToFloat<SrcDType>(
      actual_min_quantized, input_min_float, input_max_float);
  const float actual_max_float = QuantizedToFloat<SrcDType>(
      actual_max_quantized, input_min_float, input_max_float);

  RequantizeManyInNewRange<SrcDType, DstDType>(
      size, out, data, input_min_float, input_max_float,
      actual_min_float, actual_max_float);

  cudaMemcpy(outputs[0].dptr_, out,
    size * sizeof(DstDType), cudaMemcpyHostToDevice);
  cudaMemcpy(outputs[1].dptr_, &actual_min_float,
    sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(outputs[2].dptr_, &actual_max_float,
    sizeof(float), cudaMemcpyHostToDevice);
}

NNVM_REGISTER_OP(quantize_down_and_shrink_range)
.set_attr<FCompute>("FCompute<gpu>", QuantizeDownAndShrinkRangeComputeGPU);

}  // namespace op
}  // namespace mxnet
