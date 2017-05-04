/*!
 * Copyright (c) 2017 by Contributors
 * \file quantized_fully_connected.cu
 * \brief
 * \author Ziheng Jiang
*/
#include "./quantized_fully_connected-inl.h"
#include "./quantization_utils.h"
#include "../mxnet_op.h"

namespace mxnet {
namespace op {

struct QuantizedBiasAddStruct {
  MSHADOW_XINLINE static void Map(int i, float *out_min, float *out_max,
                                  const float *min_bias, const float *max_bias) {
    out_min[i] += min_bias[i];
    out_max[i] += max_bias[i];
  }
};

template<typename SrcType, typename DstType, typename CmpType>
class QuantizedFullyConnectedCublasOp : public Operator {
 public:
  explicit QuantizedFullyConnectedCublasOp(const Context& ctx,
                                   const std::vector<TShape>& in_shape,
                                   const std::vector<TShape>& out_shape,
                                   const QuantizedFullyConnectedParam& param) {
    src_type_ = mshadow::DataType<SrcType>::kCudaFlag;
    dst_type_ = mshadow::DataType<DstType>::kCudaFlag;
    cmp_type_ = mshadow::DataType<CmpType>::kCudaFlag;
    alpha_ = 1.0f;
    beta_  = 0.0f;
  }

  ~QuantizedFullyConnectedCublasOp() {
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    CHECK_EQ(in_data.size(),  9U);
    CHECK_EQ(out_data.size(), 3U);
    Stream<gpu> *s = ctx.get_stream<gpu>();
    CHECK_EQ(s->blas_handle_ownership_, Stream<gpu>::OwnHandle);
    const TBlob& data   =  in_data[0];
    const TBlob& weight =  in_data[1];
    const TBlob& bias   =  in_data[2];
    const TBlob& out    = out_data[0];
    TShape dshape = data.shape_;
    TShape wshape = weight.shape_;
    TShape oshape = out.shape_;
    // (m, n) * (k, n).T = (m, k)
    // A * B.T = C

    // row_C = col_C(T) = cublas(col_B * col_A(T)) = cublas(row_B(T), row_A)
    // row_C = col_C(T) = cublas(col_B(T) * col_A(T)) = cublas(row_B, row_A)
    int m = dshape[0], n = dshape[1], k = wshape[0];
    CUBLAS_CALL(cublasGemmEx(s->blas_handle_,
                             CUBLAS_OP_T,
                             CUBLAS_OP_N,
                             k,
                             m,
                             n,
                             &alpha_,
                             weight.dptr_,
                             src_type_,
                             n,
                             data.dptr_,
                             src_type_,
                             n,
                             &beta_,
                             out.dptr_,
                             dst_type_,
                             k,
                             cmp_type_,
                             CUBLAS_GEMM_DFALT));

    Tensor<gpu, 1, SrcType> bias_tensor = bias.get<gpu, 1, SrcType>(s);
    Tensor<gpu, 2, DstType>  out_tensor =  out.get<gpu, 2, DstType>(s);
    out_tensor += repmat(mshadow::expr::tcast<DstType>(bias_tensor), out_tensor.size(0));

    mxnet_op::Kernel<QuantizationRangeForMultiplicationStruct, gpu>::Launch(s, 1,
      out_data[1].dptr<float>(), out_data[2].dptr<float>(),
       in_data[3].dptr<float>(),  in_data[4].dptr<float>(),
       in_data[5].dptr<float>(),  in_data[6].dptr<float>());

    mxnet_op::Kernel<QuantizedBiasAddStruct, gpu>::Launch(s, 1,
      out_data[1].dptr<float>(), out_data[2].dptr<float>(),
       in_data[7].dptr<float>(),  in_data[8].dptr<float>());

  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {}


 private:
  float alpha_;
  float beta_;
  cudaDataType src_type_;
  cudaDataType dst_type_;
  cudaDataType cmp_type_;

  cudaDataType_t convertToCudaDataType(int dtype) {
    cudaDataType_t converted = CUDA_R_32F;
    MSHADOW_TYPE_SWITCH(dtype, mxDType, {
      converted = mshadow::DataType<mxDType>::kCudaFlag;
    })
    return converted;
  }
};  // class QuantizedFullyConnectedCublasOp


template<>
Operator* CreateOp<gpu>(int dtype,
                        const Context& ctx,
                        const std::vector<TShape>& in_shape,
                        const std::vector<TShape>& out_shape,
                        const QuantizedFullyConnectedParam& param) {
  Operator *op = NULL;
  op = new QuantizedFullyConnectedCublasOp<int8_t, float, float>(ctx,
    in_shape, out_shape, param);
  return op;
}

}  // namespace op
}  // namespace mxnet

