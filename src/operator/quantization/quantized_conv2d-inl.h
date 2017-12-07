/*!
 * Copyright (c) 2017 by Contributors
 * \file quantized_conv2d-inl.h
 * \brief
 * \author Ziheng Jiang
*/
#ifndef MXNET_OPERATOR_CONTRIB_QUANTIZED_CONV2D_INL_H_
#define MXNET_OPERATOR_CONTRIB_QUANTIZED_CONV2D_INL_H_
#include <mxnet/operator.h>
#include <mxnet/op_attr_types.h>
#include "../operator_common.h"

namespace mxnet {
namespace op {

namespace qconv {
enum ConvolutionOpCudnnTune {kOff, kLimited, kFastest};
}

struct QuantizedConv2DParam :
  public dmlc::Parameter<QuantizedConv2DParam> {
  TShape kernel;
  TShape stride;
  TShape pad;
  TShape dilate;
  uint32_t num_filter;
  bool no_bias;
  int layout;
  int cudnn_tune;
  bool cudnn_off;
  uint32_t num_group;
  uint64_t workspace;
  DMLC_DECLARE_PARAMETER(QuantizedConv2DParam) {
    DMLC_DECLARE_FIELD(kernel);
    DMLC_DECLARE_FIELD(stride)
    .set_default(TShape())
    .describe("conv2d stride: (h, w)");
    DMLC_DECLARE_FIELD(pad)
    .set_default(TShape())
    .describe("pad for conv2d: (h, w)");
    DMLC_DECLARE_FIELD(dilate)
    .set_default(TShape())
    .describe("convolution dilate: (h, w) or (d, h, w)");
    DMLC_DECLARE_FIELD(num_filter);
    DMLC_DECLARE_FIELD(no_bias)
    .set_default(true);
    DMLC_DECLARE_FIELD(layout)
    .add_enum("NCHW", mshadow::kNCHW)
    .add_enum("NHWC", mshadow::kNHWC)
    .set_default(mshadow::kNCHW)
    .describe("Set layout for input, output and weight. Empty for\n    "
              "default layout: NCW for 1d, NCHW for 2d and NCDHW for 3d.");
    DMLC_DECLARE_FIELD(cudnn_tune)
    .add_enum("off", qconv::kOff)
    .add_enum("limited_workspace", qconv::kLimited)
    .add_enum("fastest", qconv::kFastest)
    .set_default(qconv::kOff)
    .describe("Whether to pick convolution algo by running performance test.");
    DMLC_DECLARE_FIELD(cudnn_off)
    .set_default(false)
    .describe("Turn off cudnn for this layer.");
    DMLC_DECLARE_FIELD(num_group).set_default(1)
    .describe("Number of group partitions.");
    DMLC_DECLARE_FIELD(workspace).set_default(1024).set_range(0, 8192)
    .describe("Maximum temporary workspace allowed for convolution (MB).");
  }
};


// Declare Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(int dtype,
                   const Context& ctx,
                   const std::vector<TShape>& in_shape,
                   const std::vector<TShape>& out_shape,
                   const QuantizedConv2DParam& param);

class QuantizedConv2DProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    using namespace mshadow;
    param_.Init(kwargs);
    if (param_.stride.ndim() == 0) param_.stride = Shape2(1, 1);
    if (param_.pad.ndim() == 0)    param_.pad = Shape2(0, 0);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  std::vector<std::string> ListArguments() const override {
    if (param_.no_bias) {
      return {"data", "weight", "min_data", "max_data", "min_weight", "max_weight"};
    } else {
      return {"data", "weight", "bias", "min_data", "max_data",
        "min_weight", "max_weight", "min_bias", "max_bias"};
    }
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "min_output", "max_output"};
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    //   data[NCHW]: (batch,      channel,    in_height,     in_width)
    // kernel[NCHW]: (num_weight, channel,    weight_height, weight_width)
    //    out[NCHW]: (batch,      num_filter, out_height,    out_width)
    using namespace mshadow;
    CHECK_EQ(param_.num_group, 1U) << "quantized_conv2d only supports num_group=1 for now";
    CHECK_EQ(in_shape->size(), param_.no_bias? 6U : 9U);
    CHECK(!shape_is_none(in_shape->at(0)));
    const TShape& dshape =  in_shape->at(0);
    CHECK_EQ(dshape.ndim(), 4U);

    int N = -1, H = -1, W = -1, C = -1;
    if (param_.layout == mshadow::kNCHW) {
      N = 0, H = 2, W = 3, C = 1;
    } else if (param_.layout == mshadow::kNHWC) {
      N = 0, H = 1, W = 2, C = 3;
    } else {
      LOG(FATAL) << "not support other layout for now";
    }
    CHECK(dshape[C] % 4 == 0)
      << "for 8bit cudnn conv, the number of channel must be multiple of 4";
    CHECK(param_.num_filter % 4 == 0)
      << "for 8bit cudnn conv, the number of channel must be multiple of 4";

    TShape wshape{0, 0, 0, 0};
    wshape[N] = param_.num_filter;
    wshape[H] = param_.kernel[0];
    wshape[W] = param_.kernel[1];
    wshape[C] = dshape[C];
    SHAPE_ASSIGN_CHECK(*in_shape, 1, wshape);
    const int start = param_.no_bias? 2 : 3;
    const int end = param_.no_bias? 6 : 9;
    for (int i = start; i < end; ++i) {
      SHAPE_ASSIGN_CHECK(*in_shape, i, TShape{1});
    }
    if (!param_.no_bias) {
      SHAPE_ASSIGN_CHECK(*in_shape, 2, Shape1(param_.num_filter));
    }

    TShape oshape{1, 1, 1, 1};
    oshape[N] = dshape[N];
    oshape[C] = wshape[N];
    oshape[H] = (AddPad(dshape[H], param_.pad[0]) - wshape[H]) / param_.stride[0] + 1;
    oshape[W] = (AddPad(dshape[W], param_.pad[1]) - wshape[W]) / param_.stride[1] + 1;

    out_shape->clear();
    out_shape->push_back(oshape);
    out_shape->push_back(TShape{1});
    out_shape->push_back(TShape{1});
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_EQ(in_type->size(), param_.no_bias? 6U : 9U);
    TYPE_ASSIGN_CHECK(*in_type, 0, mshadow::kInt8);
    TYPE_ASSIGN_CHECK(*in_type, 1, mshadow::kInt8);
    if (!param_.no_bias) {
      TYPE_ASSIGN_CHECK(*in_type, 2, mshadow::kInt8);
    }

    const size_t start = param_.no_bias? 2 : 3;
    const size_t end = param_.no_bias? 6 : 9;
    for (size_t i = start; i < end; ++i) {
      TYPE_ASSIGN_CHECK(*in_type, i, mshadow::kFloat32);
    }

    out_type->clear();
    out_type->push_back(mshadow::kInt32);
    out_type->push_back(mshadow::kFloat32);
    out_type->push_back(mshadow::kFloat32);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new QuantizedConv2DProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "quantized_conv2d";
  }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return std::vector<ResourceRequest>(5, ResourceRequest::kTempSpace);
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  QuantizedConv2DParam param_;
  index_t AddPad(index_t dsize, index_t pad) const {
    return dsize + 2 * pad;
  }
};


}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_QUANTIZED_CONV2D_H_
