/*!
 * Copyright (c) 2017 by Contributors
 * \file lowbit_convolution.cu
 * \brief
 * \author Ziheng Jiang
*/
#include "./lowbit_convolution-inl.h"

namespace mxnet {
namespace op {


template<typename DType>
class LowbitConvolutionCuDNNOp : public Operator {
 public:
  explicit LowbitConvolutionCuDNNOp(const Context& ctx,
                                       const std::vector<TShape>& in_shape,
                                       const std::vector<TShape>& out_shape,
                                       const LowbitConvolutionParam& param,
                                       int cmp_type) {
    dtype_ = mshadow::DataType<DType>::kCudnnFlag;
    param_ = param;
    cmp_type_ = convertToCuDNNDataType(cmp_type);
    format_ = CUDNN_TENSOR_NCHW;
    init_temp_size_ = false;
    // 1024 MB
    workspace_limit_ = 1024;
    workspace_limit_ = (workspace_limit_ << 20) / sizeof(DType);
    InitDescriptors(ctx, in_shape, out_shape);
    SelectAlgo(ctx);
  }

  ~LowbitConvolutionCuDNNOp() {
    CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_desc_));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(data_desc_));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc_));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(fwd_conv_desc_));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(bwd_conv_desc_));
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
    CHECK_EQ(s->dnn_handle_ownership_, Stream<gpu>::OwnHandle);

    TBlob data   = in_data[0];
    TBlob filter = in_data[1];
    TBlob out    = out_data[0];
    if (!init_temp_size_) GetTempSize(ctx);
    Tensor<gpu, 1, DType> workspace =
      ctx.requested[0].get_space_typed<gpu, 1, DType>(
        mshadow::Shape1(fwd_workspace_), s);

    float alpha = 1.0f;
    float beta = 0.0f;
    CUDNN_CALL(cudnnConvolutionForward(s->dnn_handle_,
                                       &alpha,
                                       data_desc_,
                                       data.dptr_,
                                       filter_desc_,
                                       filter.dptr_,
                                       fwd_conv_desc_,
                                       fwd_algo_,
                                       workspace.dptr_,
                                       fwd_workspace_byte_,
                                       &beta,
                                       out_desc_,
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
    CHECK(in_data.size() == 2 && in_grad.size() == 2);
    Stream<gpu> *s = ctx.get_stream<gpu>();
    CHECK(param_.kernel.ndim() == 2);

    const TBlob& data    =  in_data[0];
    const TBlob& filter  =  in_data[1];
    const TBlob& ograd   = out_grad[0];
    const TBlob& gdata   =  in_grad[0];
    const TBlob& gfilter =  in_grad[1];

    Tensor<gpu, 1, DType> workspace =
      ctx.requested[0].get_space_typed<gpu, 1, DType>(
        mshadow::Shape1(bwd_workspace_), s);

    float alpha = 1.0f;
    float beta = 0.0f;
    float beta_add = 1.0f;

    if (req[1] != kNullOp) {
      CUDNN_CALL(cudnnConvolutionBackwardFilter(
        s->dnn_handle_,
        &alpha,
        data_desc_,
        data.dptr_,
        out_desc_,
        ograd.dptr_,
        bwd_conv_desc_,
        bwd_filter_algo_,
        workspace.dptr_,
        bwd_workspace_byte_,
        req[1] == kAddTo? &beta_add : &beta,
        filter_desc_,
        filter.dptr_));
    }
    if (req[0] != kNullOp) {
      CUDNN_CALL(cudnnConvolutionBackwardData(
        s->dnn_handle_,
        &alpha,
        filter_desc_,
        filter.dptr_,
        out_desc_,
        ograd.dptr_,
        bwd_conv_desc_,
        bwd_data_algo_,
        workspace.dptr_,
        bwd_workspace_byte_,
        req[0] == kAddTo? &beta_add : &beta,
        data_desc_,
        data.dptr_));
    }
  }

  void InitDescriptors(const Context& ctx,
                       const std::vector<TShape>& in_shape,
                       const std::vector<TShape>& out_shape) {
    TShape dshape =  in_shape[0];
    TShape kshape =  in_shape[1];
    TShape oshape = out_shape[0];
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&fwd_conv_desc_));
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&bwd_conv_desc_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&data_desc_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc_));
    CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_desc_));

    CUDNN_CALL(cudnnSetConvolution2dDescriptor(fwd_conv_desc_,
                                               param_.pad[0],
                                               param_.pad[1],
                                               param_.stride[0],
                                               param_.stride[1],
                                               1,
                                               1,
                                               CUDNN_CROSS_CORRELATION,
                                               cmp_type_));
    CUDNN_CALL(cudnnSetConvolution2dDescriptor(bwd_conv_desc_,
                                               param_.pad[0],
                                               param_.pad[1],
                                               param_.stride[0],
                                               param_.stride[1],
                                               1,
                                               1,
                                               CUDNN_CROSS_CORRELATION,
                                               cmp_type_));


    CUDNN_CALL(cudnnSetTensor4dDescriptor(data_desc_,
                                          format_,
                                          dtype_,
                                          dshape[0],
                                          dshape[1],
                                          dshape[2],
                                          dshape[3]));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(out_desc_,
                                          format_,
                                          dtype_,
                                          oshape[0],
                                          oshape[1],
                                          oshape[2],
                                          oshape[3]));
    CUDNN_CALL(cudnnSetFilter4dDescriptor(filter_desc_,
                                          dtype_,
                                          format_,
                                          kshape[0],
                                          kshape[1],
                                          kshape[2],
                                          kshape[3]));
  }

  void SelectAlgo(const Context& ctx) {
    Engine::VarHandle var = Engine::Get()->NewVariable();
    Engine::Get()->PushSync([=](RunContext rctx) {
      mshadow::Stream<gpu> *s = rctx.get_stream<gpu>();
      CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);
      size_t workspace_byte =
        static_cast<size_t>(workspace_limit_ * sizeof(DType));
      LOG(INFO) << "workspace_byte: " << workspace_byte;
      CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(
                 s->dnn_handle_,
                 data_desc_,
                 filter_desc_,
                 fwd_conv_desc_,
                 out_desc_,
                 CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
                 workspace_byte,
                 &fwd_algo_));
      CUDNN_CALL(cudnnGetConvolutionBackwardFilterAlgorithm(
                 s->dnn_handle_,
                 data_desc_,
                 out_desc_,
                 bwd_conv_desc_,
                 filter_desc_,
                 CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
                 workspace_byte,
                 &bwd_filter_algo_));
      CUDNN_CALL(cudnnGetConvolutionBackwardDataAlgorithm(
                 s->dnn_handle_,
                 filter_desc_,
                 out_desc_,
                 bwd_conv_desc_,
                 data_desc_,
                 CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
                 workspace_byte,
                 &bwd_data_algo_));
    }, ctx, {}, {var});
    Engine::Get()->WaitForVar(var);
    Engine::Get()->DeleteVariable([](RunContext rctx) {}, ctx, var);
  }

  void GetTempSize(const OpContext& ctx) {
    CHECK(!init_temp_size_) << "GetTempSize should only be called once.";
    mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
    CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(s->dnn_handle_,
                                                       data_desc_,
                                                       filter_desc_,
                                                       fwd_conv_desc_,
                                                       out_desc_,
                                                       fwd_algo_,
                                                       &fwd_workspace_byte_));
    size_t bwd_data_size = 0, bwd_filter_size = 0;
    CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
                 s->dnn_handle_,
                 filter_desc_,
                 out_desc_,
                 bwd_conv_desc_,
                 data_desc_,
                 bwd_data_algo_,
                 &bwd_data_size));
    CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
                 s->dnn_handle_,
                 data_desc_,
                 out_desc_,
                 bwd_conv_desc_,
                 filter_desc_,
                 bwd_filter_algo_,
                 &bwd_filter_size));
    bwd_workspace_byte_ = std::max(bwd_data_size, bwd_filter_size);

    fwd_workspace_ = fwd_workspace_byte_ / sizeof(DType) + 1;
    bwd_workspace_ = bwd_workspace_byte_ / sizeof(DType) + 1;
    init_temp_size_ = true;
  }


 private:
  bool init_temp_size_ = false;
  size_t workspace_limit_;
  LowbitConvolutionParam param_;
  size_t fwd_workspace_;
  size_t fwd_workspace_byte_;
  size_t bwd_workspace_;
  size_t bwd_workspace_byte_;
  cudnnDataType_t dtype_;
  cudnnDataType_t cmp_type_;
  cudnnTensorFormat_t format_;
  cudnnConvolutionDescriptor_t fwd_conv_desc_;
  cudnnConvolutionDescriptor_t bwd_conv_desc_;
  cudnnTensorDescriptor_t data_desc_;
  cudnnFilterDescriptor_t filter_desc_;
  cudnnTensorDescriptor_t out_desc_;
  cudnnConvolutionFwdAlgo_t fwd_algo_;
  cudnnConvolutionBwdDataAlgo_t   bwd_data_algo_;
  cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo_;

  cudnnDataType_t convertToCuDNNDataType(int dtype) {
    cudnnDataType_t converted = CUDNN_DATA_FLOAT;
    // The following will always assign to `converted` or throw an exception.
    MSHADOW_TYPE_SWITCH(dtype, mxDType, {
      converted = mshadow::DataType<mxDType>::kCudnnFlag;
    })
    return converted;
  }

};  // class LowbitReluCuDNNOp


template<>
Operator* CreateOp<gpu>(int dtype,
                        const Context& ctx,
                        const std::vector<TShape>& in_shape,
                        const std::vector<TShape>& out_shape,
                        const LowbitConvolutionParam& param) {
  Operator *op = NULL;
  LOG(INFO) << "dtype: " << dtype << ", "<< (dtype == mshadow::kInt8);
  int cmp_type = (dtype == mshadow::kInt8) ? mshadow::kInt32 : dtype;
  LOG(INFO) << "cmp_type: " << cmp_type << ", "<< (cmp_type == mshadow::kInt32);
  MSHADOW_TYPE_SWITCH(dtype, DType, {
    op = new LowbitConvolutionCuDNNOp<DType>(ctx,
      in_shape, out_shape, param, cmp_type);
  })
  return op;
}

}  // namespace op
}  // namespace mxnet

