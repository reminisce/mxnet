/*!
 * Copyright (c) 2017 by Contributors
 * \file lowbit_fully_connected.cu
 * \brief
 * \author Ziheng Jiang
*/
#include "./lowbit_fully_connected-inl.h"

namespace mxnet {
namespace op {

template<typename DType>
class LowbitFullyConnectedCublasOp : public Operator {
 public:
  explicit LowbitFullyConnectedCublasOp(const Context& ctx,
                                       const std::vector<TShape>& in_shape,
                                       const std::vector<TShape>& out_shape,
                                       const LowbitFullyConnectedParam& param,
                                       int cmp_type) {
    dtype_ = mshadow::DataType<DType>::kCudaFlag;
    cmp_type_ = convertToCudaDataType(cmp_type);
    alpha_ = 1.0f;
    beta_  = 0.0f;
  }

  ~LowbitFullyConnectedCublasOp() {
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    CHECK_EQ(in_data.size(), 2U);
    CHECK_EQ(out_data.size(), 1U);
    Stream<gpu> *s = ctx.get_stream<gpu>();
    CHECK_EQ(s->blas_handle_ownership_, Stream<gpu>::OwnHandle);
    const TBlob& data   =  in_data[0];
    const TBlob& weight =  in_data[1];
    const TBlob& out    = out_data[0];
    TShape dshape = data.shape_;
    TShape wshape = weight.shape_;
    TShape oshape = out.shape_;

    CUBLAS_CALL(cublasGemmEx(s->blas_handle_,
                             CUBLAS_OP_N,
                             CUBLAS_OP_T,
                             dshape[0],
                             dshape[1],
                             wshape[0],
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
};  // class LowbitFullyConnectedCublasOp


template<>
Operator* CreateOp<gpu>(int dtype,
                        const Context& ctx,
                        const std::vector<TShape>& in_shape,
                        const std::vector<TShape>& out_shape,
                        const LowbitFullyConnectedParam& param) {
  Operator *op = NULL;
  int cmp_type = (dtype == mshadow::kInt8) ? mshadow::kInt32 : dtype;
  MSHADOW_TYPE_SWITCH(dtype, DType, {
    op = new LowbitFullyConnectedCublasOp<DType>(ctx,
      in_shape, out_shape, param, cmp_type);
  })
  return op;
}

}  // namespace op
}  // namespace mxnet

