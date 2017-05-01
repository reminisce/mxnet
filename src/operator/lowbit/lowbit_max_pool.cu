/*!
 * Copyright (c) 2017 by Contributors
 * \file lowbit_max_pool.cu
 * \brief
 * \author Ziheng Jiang
*/
#include <vector>
#include "./lowbit_max_pool-inl.h"

namespace mxnet {
namespace op {

template<typename DType>
class LowbitMaxPoolCuDNNOp : public Operator {
 public:
  explicit LowbitMaxPoolCuDNNOp(LowbitMaxPoolParam p) {
    param_ = p;
    init_cudnn_ = false;
    alpha_ = 1.0f;
    beta_  = 0.0f;
    dtype_ = mshadow::DataType<DType>::kCudnnFlag;
    mode_ = CUDNN_POOLING_MAX;
    nan_prop_ = CUDNN_NOT_PROPAGATE_NAN;
  }

  ~LowbitMaxPoolCuDNNOp() {
    if (init_cudnn_) {
      CUDNN_CALL(cudnnDestroyTensorDescriptor(in_desc_));
      CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc_));
      CUDNN_CALL(cudnnDestroyPoolingDescriptor(pool_desc_));
    }
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    CHECK_EQ(in_data.size(), 1U);
    CHECK_EQ(out_data.size(), 1U);
    Stream<gpu> *s = ctx.get_stream<gpu>();
    CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);
    CHECK(param_.kernel.ndim() == 2) << "Only support 2D pooling";
    if (!init_cudnn_) this->Init(s, in_data, out_data);
    CUDNN_CALL(cudnnPoolingForward(s->dnn_handle_,
                                   pool_desc_,
                                   &alpha_,
                                   in_desc_,
                                   in_data[0].dptr_,
                                   &beta_,
                                   out_desc_,
                                   out_data[0].dptr_));
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    CHECK_EQ(out_grad.size(), 1U);
    CHECK_EQ(in_data.size(), 1U);
    CHECK_EQ(out_data.size(), 1U);
    CHECK_EQ(req.size(), 1U);
    CHECK_EQ(in_grad.size(), 1U);
    Stream<gpu> *s = ctx.get_stream<gpu>();
    CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);
    CHECK(param_.kernel.ndim() == 2) << "Only support 2D pooling";
    const TBlob& data  =  in_data[0];
    const TBlob& out   = out_data[0];
    const TBlob& ograd = out_grad[0];
    const TBlob& igrad =  in_grad[0];
    CUDNN_CALL(cudnnPoolingBackward(s->dnn_handle_,
                                    pool_desc_,
                                    &alpha_,
                                    out_desc_,
                                    out.dptr_,
                                    out_desc_,
                                    ograd.dptr_,
                                    in_desc_,
                                    data.dptr_,
                                    &beta_,
                                    in_desc_,
                                    igrad.dptr_));
  }

 private:
  inline void Init(mshadow::Stream<gpu> *s,
                   const std::vector<TBlob> &in_data,
                   const std::vector<TBlob> &out_data) {
    using namespace mshadow;
    CHECK(!init_cudnn_) << "Init should only be called when init_cudnn is false";
    CHECK_EQ(in_data.size(), 1U);
    CHECK_EQ(out_data.size(), 1U);
    CHECK(param_.kernel.ndim() == 2) << "only support 2d pooling";
    const TBlob& data = in_data[0];
    const TBlob& out  = out_data[0];
    TShape dshape = data.shape_;
    CUDNN_CALL(cudnnCreatePoolingDescriptor(&pool_desc_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc_));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(in_desc_,
                                          CUDNN_TENSOR_NCHW,
                                          dtype_,
                                          dshape[0],
                                          dshape[1],
                                          dshape[2],
                                          dshape[3]));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(out_desc_,
                                          CUDNN_TENSOR_NCHW,
                                          dtype_,
                                          dshape[0],
                                          dshape[1],
                                          dshape[2],
                                          dshape[3]));
    CUDNN_CALL(cudnnSetPooling2dDescriptor(
      pool_desc_,
      mode_,
      nan_prop_,
      param_.kernel[0],
      param_.kernel[1],
      param_.pad[0],
      param_.pad[1],
      param_.stride[0],
      param_.stride[1]));
  }
  bool init_cudnn_;
  float alpha_;
  float beta_;
  cudnnDataType_t dtype_;
  cudnnHandle_t handle_;
  cudnnPoolingMode_t mode_;
  cudnnTensorDescriptor_t in_desc_;
  cudnnTensorDescriptor_t out_desc_;
  cudnnPoolingDescriptor_t pool_desc_;
  cudnnNanPropagation_t nan_prop_;
  LowbitMaxPoolParam param_;
};  // class LowbitMaxPoolCuDNNOp

template<>
Operator *CreateOp<gpu>(LowbitMaxPoolParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_TYPE_SWITCH(dtype, DType, {
    op = new LowbitMaxPoolCuDNNOp<DType>(param);
  });
  return op;
}

}  // namespace op
}  // namespace mxnet

