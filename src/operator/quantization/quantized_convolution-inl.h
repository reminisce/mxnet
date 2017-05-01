/*!
 * Copyright (c) 2017 by Contributors
 * \file quantized_convolution-inl.h
 * \brief
 * \author Ziheng Jiang
*/

#ifndef MXNET_OPERATOR_CONTRIB_QUANTIZED_CONVOLUTION_INL_H_
#define MXNET_OPERATOR_CONTRIB_QUANTIZED_CONVOLUTION_INL_H_
#include <mxnet/operator.h>
#include "../operator_common.h"

namespace mxnet {
namespace op {

struct QuantizedConvolutionParam :
  public dmlc::Parameter<QuantizedConvolutionParam> {
  TShape stride;
  TShape pad;
  DMLC_DECLARE_PARAMETER(QuantizedConvolutionParam) {
    DMLC_DECLARE_FIELD(stride).set_default(TShape())
    .describe("convolution stride: (h, w)");
    DMLC_DECLARE_FIELD(pad).set_default(TShape())
    .describe("pad for convolution: (h, w)");
  }
};


// Declare Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(int dtype,
                   const Context& ctx,
                   const std::vector<TShape>& in_shape,
                   const std::vector<TShape>& out_shape,
                   const QuantizedConvolutionParam& param);

class QuantizedConvolutionProp : public OperatorProperty {
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
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 6U);
    CHECK_EQ(out_shape->size(), 3U);
    for (int i = 0; i < 2; ++i) {
      CHECK(!shape_is_none(in_shape->at(0)));
    }
    for (int i = 2; i < 6; ++i) {
      CHECK(shape_is_scalar(in_shape->at(i)));
    }
    const TShape& dshape =  in_shape->at(0);
    const TShape& fshape =  in_shape->at(1);
    TShape& oshape = out_shape->at(0);

    CHECK_EQ(dshape.ndim(), 4U); // batch_size, in_filter, h, w
    CHECK_EQ(fshape.ndim(), 4U); // out_filter, in_filter, filter_h, filter_w

    oshape[0] = dshape[0];
    oshape[1] = fshape[0];
    oshape[2] = (AddPad(dshape[2], param_.pad[0])) / param_.stride[0] + 1;
    oshape[3] = (AddPad(dshape[3], param_.pad[1])) / param_.stride[1] + 1;

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
    for (size_t i = 0; i < 2; ++i) {
      CHECK_EQ((*in_type)[i], mshadow::kInt8)
        << "`quantized_matmul` only supports int8 input for now";
    }
    for (size_t i = 2; i < 6; ++i) {
      CHECK_EQ((*in_type)[1], mshadow::kFloat32)
        << "the " << i << "th input of `quantized_matmul` should"
        << "be a tensor with type of float32";
    }

    out_type->clear();
    out_type->push_back(mshadow::kInt8);
    out_type->push_back(mshadow::kFloat32);
    out_type->push_back(mshadow::kFloat32);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new QuantizedConvolutionProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "quantized_convolution";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[0], in_data[0], in_data[1]};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  QuantizedConvolutionParam param_;
  index_t AddPad(index_t dsize, index_t pad) const {
    return dsize + 2 * pad;
  }
};


}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_QUANTIZED_CONVOLUTION_H_
