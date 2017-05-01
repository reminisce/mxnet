/*!
 * Copyright (c) 2017 by Contributors
 * \file quantized_matmul.cu
 * \brief
 * \author Ziheng Jiang
*/
#include "./quantized_matmul-inl.h"
#include "./quantization_utils.h"
#include "../mxnet_op.h"

namespace mxnet {
namespace op {

template<typename DType, typename CmpType>
class QuantizedMatmulCublasOp : public Operator {
 public:
  explicit QuantizedMatmulCublasOp(const Context& ctx,
                                   const std::vector<TShape>& in_shape,
                                   const std::vector<TShape>& out_shape,
                                   const QuantizedMatmulParam& param) {
    dtype_    = mshadow::DataType<DType>::kCudaFlag;
    cmp_type_ = mshadow::DataType<CmpType>::kCudaFlag;
    alpha_ = 1.0f;
    beta_  = 0.0f;
  }

  ~QuantizedMatmulCublasOp() {
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    CHECK_EQ(in_data.size(), 6U);
    CHECK_EQ(out_data.size(), 3U);
    Stream<gpu> *s = ctx.get_stream<gpu>();
    CHECK_EQ(s->blas_handle_ownership_, Stream<gpu>::OwnHandle);
    const TBlob& data   =  in_data[0];
    const TBlob& weight =  in_data[1];
    const TBlob& out    = out_data[0];
    TShape dshape = data.shape_;
    TShape wshape = weight.shape_;
    TShape oshape = out.shape_;

    int m = dshape[0], n = dshape[1], k = wshape[1];
    CUBLAS_CALL(cublasGemmEx(s->blas_handle_,
                             CUBLAS_OP_N,
                             CUBLAS_OP_N,
                             m,
                             n,
                             k,
                             &alpha_,
                             data.dptr_,
                             dtype_,
                             dshape[1],
                             weight.dptr_,
                             dtype_,
                             wshape[1],
                             &beta_,
                             out.dptr_,
                             cmp_type_,
                             oshape[1],
                             cmp_type_,
                             CUBLAS_GEMM_DFALT));

    mxnet_op::Kernel<quantization_range_for_multiplication, gpu>::Launch(s, 1,
      out_data[1].dptr<float>(), out_data[2].dptr<float>(),
       in_data[3].dptr<float>(),  in_data[4].dptr<float>(),
       in_data[5].dptr<float>(),  in_data[6].dptr<float>());
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
  cudaDataType dtype_;
  cudaDataType cmp_type_;

  cudaDataType_t convertToCudaDataType(int dtype) {
    cudaDataType_t converted = CUDA_R_32F;
    MSHADOW_TYPE_SWITCH(dtype, mxDType, {
      converted = mshadow::DataType<mxDType>::kCudaFlag;
    })
    return converted;
  }
};  // class QuantizedMatmulCublasOp


template<>
Operator* CreateOp<gpu>(int dtype,
                        const Context& ctx,
                        const std::vector<TShape>& in_shape,
                        const std::vector<TShape>& out_shape,
                        const QuantizedMatmulParam& param) {
  Operator *op = NULL;
  op = new QuantizedMatmulCublasOp<int8_t, int32_t>(ctx,
    in_shape, out_shape, param);
  return op;
}

}  // namespace op
}  // namespace mxnet

