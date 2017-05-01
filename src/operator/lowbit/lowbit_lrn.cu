/*!
 * Copyright (c) 2017 by Contributors
 * \file lowbit_lrn.cu
 * \brief
 * \author Ziheng Jiang
*/

#include "./lowbit_lrn-inl.h"

namespace mxnet {
namespace op {

template<typename DType>
class LowbitLRNCuDNNOp : public Operator {
 public:
  explicit LowbitLRNCuDNNOp(LowbitLRNParam param) {
    param_ = param;
    init_cudnn_ = false;
    alpha_ = 1.0f;
    beta_  = 0.0f;
    dtype_ = mshadow::DataType<DType>::kCudnnFlag;
  }

  ~LowbitLRNCuDNNOp() {
    if (init_cudnn_) {
      CUDNN_CALL(cudnnDestroyLRNDescriptor(lrn_desc_));
      CUDNN_CALL(cudnnDestroyTensorDescriptor(shape_desc_));
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
    const TBlob& data = in_data[0];
    const TBlob& out = out_data[0];
    if (!init_cudnn_) this->Init(s, in_data, out_data);
    CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);
    CUDNN_CALL(cudnnLRNCrossChannelForward(s->dnn_handle_,
                                           lrn_desc_,
                                           CUDNN_LRN_CROSS_CHANNEL_DIM1,
                                           &alpha_,
                                           shape_desc_,
                                           data.dptr_,
                                           &beta_,
                                           shape_desc_,
                                           out.dptr_));
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
    CHECK_EQ(out_data.size(), 2U);
    CHECK_EQ(req.size(), 1U);
    CHECK_EQ(in_grad.size(), 1U);
    Stream<gpu> *s = ctx.get_stream<gpu>();

    const TBlob& data  =  in_data[0];
    const TBlob& out   = out_data[0];
    const TBlob& ograd = out_grad[0];
    const TBlob& igrad =  in_grad[0];
    CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);
    CUDNN_CALL(cudnnLRNCrossChannelBackward(s->dnn_handle_,
                                            lrn_desc_,
                                            CUDNN_LRN_CROSS_CHANNEL_DIM1,
                                            &alpha_,
                                            shape_desc_,
                                            out.dptr_,
                                            shape_desc_,
                                            ograd.dptr_,
                                            shape_desc_,
                                            data.dptr_,
                                            &beta_,
                                            shape_desc_,
                                            igrad.dptr_));
  }

 private:
  inline void Init(mshadow::Stream<gpu> *s,
                   const std::vector<TBlob> &in_data,
                   const std::vector<TBlob> &out_data) {
    using namespace mshadow;
    CHECK(!init_cudnn_) << "Init should only be called when not initialized";
    init_cudnn_ = true;
    const TBlob& data = in_data[0];
    CUDNN_CALL(cudnnCreateLRNDescriptor(&lrn_desc_));
    CUDNN_CALL(cudnnSetLRNDescriptor(lrn_desc_,
                                     param_.nsize,
                                     param_.alpha,
                                     param_.beta,
                                     param_.knorm))
    CUDNN_CALL(cudnnCreateTensorDescriptor(&shape_desc_));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(shape_desc_,
                                          CUDNN_TENSOR_NCHW,
                                          dtype_,
                                          data.shape_[0],
                                          data.shape_[1],
                                          data.shape_[2],
                                          data.shape_[3]));
  }
  bool init_cudnn_;
  float alpha_;
  float beta_;
  LowbitLRNParam param_;
  cudnnDataType_t dtype_;
  cudnnLRNDescriptor_t lrn_desc_;
  cudnnTensorDescriptor_t shape_desc_;
};  // class CuDNNLocalResponseNormOp


template<>
Operator* CreateOp<gpu>(LowbitLRNParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_TYPE_SWITCH(dtype, DType, {
    op = new LowbitLRNCuDNNOp<DType>(param);
  })
  return op;
}

}  // namespace op
}  // namespace mxnet


