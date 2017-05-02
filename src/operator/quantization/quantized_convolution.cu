/*!
 * Copyright (c) 2017 by Contributors
 * \file quantized_convolution.cu
 * \brief
 * \author Ziheng Jiang
*/
#include "./quantized_convolution-inl.h"
#include "./quantization_utils.h"

namespace mxnet {
namespace op {

template<typename SrcType, typename DstType, typename CmpType>
class QuantizedConvolutionCuDNNOp : public Operator {
 public:
  explicit QuantizedConvolutionCuDNNOp(const Context& ctx,
                                       const std::vector<TShape>& in_shape,
                                       const std::vector<TShape>& out_shape,
                                       const QuantizedConvolutionParam& param) {
    param_ = param;
    src_type_ = mshadow::DataType<SrcType>::kCudnnFlag;
    cmp_type_ = mshadow::DataType<CmpType>::kCudnnFlag;
    algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    format_ = CUDNN_TENSOR_NHWC;
    init_temp_size_ = false;
    // 1024 MB
    workspace_limit_ = 1024;
    workspace_limit_ = (workspace_limit_ << 20) / sizeof(SrcType);
    InitDescriptors(ctx, in_shape, out_shape);
  }

  ~QuantizedConvolutionCuDNNOp() {
    CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_desc_));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(data_desc_));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc_));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc_));
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
    CHECK_EQ(s->dnn_handle_ownership_, Stream<gpu>::OwnHandle);

    TBlob data   = in_data[0];
    TBlob filter = in_data[1];
    TBlob out    = out_data[0];
    if (!init_temp_size_) GetTempSize(ctx);
    LOG(INFO) << "Resource Request: " << workspace_;
    Tensor<gpu, 1, SrcType> workspace =
      ctx.requested[0].get_space_typed<gpu, 1, SrcType>(mshadow::Shape1(workspace_), s);

    float alpha = 1.0f;
    float beta = 0.0f;
    LOG(INFO) << "CuDNN Forward";
    CUDNN_CALL(cudnnConvolutionForward(s->dnn_handle_,
                                       &alpha,
                                       data_desc_,
                                       data.dptr_,
                                       filter_desc_,
                                       filter.dptr_,
                                       conv_desc_,
                                       algo_,
                                       workspace.dptr_,
                                       workspace_byte_,
                                       &beta,
                                       out_desc_,
                                       out.dptr_));

    mxnet_op::Kernel<quantization_range_for_multiplication, gpu>::Launch(s, 1,
      out_data[1].dptr<float>(), out_data[2].dptr<float>(),
       in_data[2].dptr<float>(),  in_data[3].dptr<float>(),
       in_data[4].dptr<float>(),  in_data[5].dptr<float>());
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {}


  void InitDescriptors(const Context& ctx,
                       const std::vector<TShape>& in_shape,
                       const std::vector<TShape>& out_shape) {
    TShape dshape =  in_shape[0];
    TShape kshape =  in_shape[1];
    TShape oshape = out_shape[0];
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&data_desc_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc_));
    CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_desc_));

    CUDNN_CALL(cudnnSetConvolution2dDescriptor(conv_desc_,
                                               param_.pad[0],
                                               param_.pad[1],
                                               param_.stride[0],
                                               param_.stride[1],
                                               1,
                                               1,
                                               CUDNN_CROSS_CORRELATION,
                                               cmp_type_));

    LOG(INFO) << "dshape: " << dshape
      << ", kshape: " << kshape
      << ", oshape: " << oshape;

    CUDNN_CALL(cudnnSetTensor4dDescriptor(data_desc_,
                                          format_,
                                          src_type_,
                                          dshape[0],
                                          dshape[3],
                                          dshape[1],
                                          dshape[2]));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(out_desc_,
                                          format_,
                                          src_type_,
                                          oshape[0],
                                          oshape[3],
                                          oshape[1],
                                          oshape[2]));
    // input:  [NHWC](batch, in_height, in_width, in_channels)
    // filter: [HWNC](out_channels, filter_height, filter_width, in_channels)
    // output: [NHWC](batch, out_height, out_width, out_channels)
    CUDNN_CALL(cudnnSetFilter4dDescriptor(filter_desc_,
                                          src_type_,
                                          format_,
                                          kshape[0],
                                          kshape[3],
                                          kshape[1],
                                          kshape[2]));
  }

  void GetTempSize(const OpContext& ctx) {
    CHECK(!init_temp_size_)
      << "GetTempSize should only be called once.";
    mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
    CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(s->dnn_handle_,
                                                       data_desc_,
                                                       filter_desc_,
                                                       conv_desc_,
                                                       out_desc_,
                                                       algo_,
                                                       &workspace_byte_));
    workspace_ = workspace_byte_ / sizeof(SrcType) + 1;
    init_temp_size_ = true;
    LOG(INFO) << "GetWorkspaceSize Done";
  }


 private:
  bool init_temp_size_ = false;
  size_t workspace_limit_;
  QuantizedConvolutionParam param_;
  size_t workspace_;
  size_t workspace_byte_;
  cudnnDataType_t src_type_;
  cudnnDataType_t cmp_type_;
  cudnnTensorFormat_t format_;
  cudnnConvolutionDescriptor_t conv_desc_;
  cudnnTensorDescriptor_t data_desc_;
  cudnnFilterDescriptor_t filter_desc_;
  cudnnTensorDescriptor_t out_desc_;
  cudnnConvolutionFwdAlgo_t algo_;

  cudnnDataType_t convertToCuDNNDataType(int dtype) {
    cudnnDataType_t converted = CUDNN_DATA_FLOAT;
    // The following will always assign to `converted` or throw an exception.
    MSHADOW_TYPE_SWITCH(dtype, mxDType, {
      converted = mshadow::DataType<mxDType>::kCudnnFlag;
    })
    return converted;
  }

};  // class QuantizedReluCuDNNOp


template<>
Operator* CreateOp<gpu>(int dtype,
                        const Context& ctx,
                        const std::vector<TShape>& in_shape,
                        const std::vector<TShape>& out_shape,
                        const QuantizedConvolutionParam& param) {
  Operator *op = NULL;
  op = new QuantizedConvolutionCuDNNOp<int8_t, int8_t, int32_t>(ctx,
    in_shape, out_shape, param);
  return op;
}

}  // namespace op
}  // namespace mxnet

