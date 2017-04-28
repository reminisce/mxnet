/*!
 * Copyright (c) 2015 by Contributors
 * \file activation.cu
 * \brief
 * \author Bing Xu
*/
// #include <mshadow/base.h>
// #include <mxnet/base.h>
// #include <mxnet/operator.h>
#include "./quantized_relu-inl.h"

namespace mxnet {
namespace op {


template<typename DType>
class QuantizedReluCuDNNOp : public Operator {
 public:
  explicit QuantizedReluCuDNNOp() {
    init_cudnn_ = false;
    dtype_ = mshadow::DataType<DType>::kCudnnFlag;
    mode_ = CUDNN_ACTIVATION_RELU;
    nan_prop_ = CUDNN_NOT_PROPAGATE_NAN;
    CHECK_EQ(cudnnCreateActivationDescriptor(&desc_),
             CUDNN_STATUS_SUCCESS);
    CHECK_EQ(cudnnSetActivationDescriptor(desc_, mode_, nan_prop_, relu_ceil_),
             CUDNN_STATUS_SUCCESS);
  }

  ~QuantizedReluCuDNNOp() {
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
    float alpha = 1.0f;
    float beta = 0.0f;
    CHECK_EQ(s->dnn_handle_ownership_, Stream<gpu>::OwnHandle);
    if (!init_cudnn_) {
      init_cudnn_ = true;
      CHECK_EQ(cudnnCreateTensorDescriptor(&shape_desc_), CUDNN_STATUS_SUCCESS);
      CHECK_EQ(cudnnSetTensor4dDescriptor(shape_desc_,
                                          CUDNN_TENSOR_NCHW,
                                          dtype_,
                                          shape[0],
                                          shape[1],
                                          shape[2],
                                          shape[3]), CUDNN_STATUS_SUCCESS);
    }
    CHECK_EQ(cudnnActivationForward(s->dnn_handle_,
                                    desc_,
                                    &alpha,
                                    shape_desc_,
                                    data.dptr_,
                                    &beta,
                                    shape_desc_,
                                    out.dptr_), CUDNN_STATUS_SUCCESS);
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {}

 private:
  bool init_cudnn_;
  cudnnDataType_t dtype_;
  cudnnActivationMode_t mode_;
  cudnnTensorDescriptor_t shape_desc_;
  cudnnActivationDescriptor_t desc_;
  cudnnNanPropagation_t nan_prop_;
  double relu_ceil_;
};  // class QuantizedReluCuDNNOp


template<>
Operator *CreateOp<gpu>(int dtype) {
  Operator *op = NULL;
  MSHADOW_TYPE_SWITCH(dtype, DType, {
    op = new QuantizedReluCuDNNOp<DType>();
  })
  return op;
}
}  // namespace op
}  // namespace mxnet

