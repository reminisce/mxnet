/*!
 * Copyright (c) 2017 by Contributors
 * \file lowbit_relu.cu
 * \brief
 * \author Ziheng Jiang
*/
#include "./lowbit_relu-inl.h"

namespace mxnet {
namespace op {

template<typename DType>
class LowbitReluCuDNNOp : public Operator {
 public:
  explicit LowbitReluCuDNNOp() {
    init_cudnn_ = false;
    dtype_ = mshadow::DataType<DType>::kCudnnFlag;
    mode_  = CUDNN_ACTIVATION_RELU;
    nan_prop_ = CUDNN_NOT_PROPAGATE_NAN;
    alpha_ = 1.0f;
    beta_ = 0.0f;
  }

  ~LowbitReluCuDNNOp() {
    if (init_cudnn_) {
      CHECK_EQ(cudnnDestroyTensorDescriptor(shape_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnDestroyActivationDescriptor(desc_), CUDNN_STATUS_SUCCESS);
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

    TBlob data = in_data[0], out = out_data[0];
    if (data.shape_.ndim() > 4) {
      LOG(FATAL) << "Not support yet";
    }
    Shape<4> shape = Shape4(1, 1, 1, 1);
    for (size_t i = 0; i < data.shape_.ndim(); ++i) {
      shape[i] = data.shape_[i];
    }
    CHECK_EQ(s->dnn_handle_ownership_, Stream<gpu>::OwnHandle);
    if (!init_cudnn_) {
      InitDescriptors(shape);
      init_cudnn_ = true;
    }
    CUDNN_CALL(cudnnActivationForward(s->dnn_handle_,
                                      desc_,
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
    CHECK_EQ(out_data.size(), 1U);
    CHECK_EQ(req.size(), 1U);
    CHECK_EQ(in_grad.size(), 1U);
    Stream<gpu> *s = ctx.get_stream<gpu>();

    const TBlob& data = in_data[0];
    const TBlob& out  = out_data[0];
    const TBlob& igrad = in_grad[0];
    const TBlob& ograd = out_grad[0];

    CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);
    CUDNN_CALL(cudnnActivationBackward(s->dnn_handle_,
                                       desc_,
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
  bool init_cudnn_;
  cudnnDataType_t dtype_;
  cudnnActivationMode_t mode_;
  cudnnTensorDescriptor_t shape_desc_;
  cudnnActivationDescriptor_t desc_;
  cudnnNanPropagation_t nan_prop_;
  double relu_ceil_;
  float alpha_ = 1.0f;
  float beta_ = 0.0f;

  void InitDescriptors(TShape shape) {
    CHECK(!init_cudnn_);
    CUDNN_CALL(cudnnCreateActivationDescriptor(&desc_));
    CUDNN_CALL(cudnnSetActivationDescriptor(desc_, mode_, nan_prop_, relu_ceil_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&shape_desc_));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(shape_desc_,
                                          CUDNN_TENSOR_NCHW,
                                          dtype_,
                                          shape[0],
                                          shape[1],
                                          shape[2],
                                          shape[3]));
  }
};  // class LowbitReluCuDNNOp


template<>
Operator *CreateLowbitReluOp<gpu>(int dtype) {
  Operator *op = NULL;
  MSHADOW_TYPE_SWITCH(dtype, DType, {
    op = new LowbitReluCuDNNOp<DType>();
  })
  return op;
}
}  // namespace op
}  // namespace mxnet

