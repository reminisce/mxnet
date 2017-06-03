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

struct QuantizedConv2DParam :
  public dmlc::Parameter<QuantizedConv2DParam> {
  TShape kernel;
  TShape stride;
  TShape pad;
  uint32_t num_filter;
  bool no_bias;
  DMLC_DECLARE_PARAMETER(QuantizedConv2DParam) {
    DMLC_DECLARE_FIELD(kernel);
    DMLC_DECLARE_FIELD(stride)
    .set_default(TShape())
    .describe("conv2d stride: (h, w)");
    DMLC_DECLARE_FIELD(pad)
    .set_default(TShape())
    .describe("pad for conv2d: (h, w)");
    DMLC_DECLARE_FIELD(num_filter);
    DMLC_DECLARE_FIELD(no_bias)
    .set_default(true);
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

  std::vector<std::string> ListArguments() const {
    return {"data", "filter", "min_data", "max_data", "min_filter", "max_filter"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"out", "min_out", "max_out"};
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    //   data[NCHW]: (batch,      channel,    in_height,     in_width)
    // kernel[NCHW]: (num_filter, channel,    filter_height, filter_width)
    //    out[NCHW]: (batch,      num_filter, out_height,    out_width)
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 6U);
    CHECK(!shape_is_none(in_shape->at(0)));
    const TShape& dshape =  in_shape->at(0);
    CHECK_EQ(dshape.ndim(), 4U);
    CHECK(dshape[1] % 4 == 0)
      << "for 8bit cudnn conv, the number of channel must be multiple of 4";
    CHECK(param_.num_filter % 4 == 0)
      << "for 8bit cudnn conv, the number of channel must be multiple of 4";

    TShape fshape = Shape4(param_.num_filter, dshape[1], param_.kernel[0], param_.kernel[1]);
    SHAPE_ASSIGN_CHECK(*in_shape, 1, fshape);
    for (int i = 2; i < 6; ++i) {
      SHAPE_ASSIGN_CHECK(*in_shape, i, TShape{1});
    }

    TShape oshape{1, 1, 1, 1};
    oshape[0] = dshape[0];
    oshape[1] = fshape[0];
    oshape[2] = (AddPad(dshape[2], param_.pad[0]) - fshape[2]) / param_.stride[0] + 1;
    oshape[3] = (AddPad(dshape[3], param_.pad[1]) - fshape[3]) / param_.stride[1] + 1;

    out_shape->clear();
    out_shape->push_back(oshape);
    out_shape->push_back(TShape{1});
    out_shape->push_back(TShape{1});
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_EQ(in_type->size(), 6U);
    CHECK_EQ((*in_type)[0], mshadow::kInt8)
      << "`quantized_conv2d` only supports int8 input for now";
    TYPE_ASSIGN_CHECK(*in_type, 1, mshadow::kInt8);

    for (size_t i = 2; i < 6; ++i) {
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
