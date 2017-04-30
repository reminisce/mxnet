/*!
 * Copyright (c) 2017 by Contributors
 * \file quantized_lrn-inl.h
 * \brief
 * \author Ziheng Jiang
*/
#ifndef MXNET_OPERATOR_CONTRIB_QUANTIZED_LRN_INL_H_
#define MXNET_OPERATOR_CONTRIB_QUANTIZED_LRN_INL_H_
#include <mxnet/operator.h>
#include "../operator_common.h"

namespace mxnet {
namespace op {

struct QuantizedLRNParam : public dmlc::Parameter<QuantizedLRNParam> {
  float alpha;
  float beta;
  float knorm;
  uint32_t nsize;
  DMLC_DECLARE_PARAMETER(QuantizedLRNParam) {
    DMLC_DECLARE_FIELD(alpha).set_default(1e-4f)
    .describe("The variance scaling parameter :math:`\alpha` in the LRN expression.");
    DMLC_DECLARE_FIELD(beta).set_default(0.75f)
    .describe("The power parameter :math:`\beta` in the LRN expression.");
    DMLC_DECLARE_FIELD(knorm).set_default(2.0f)
    .describe("The parameter :math:`k` in the LRN expression.");
    DMLC_DECLARE_FIELD(nsize)
    .describe("normalization window width in elements.");
  }
};  // struct QuantizedLRNParam

template<typename xpu>
Operator *CreateOp(QuantizedLRNParam param, int dtype);

class QuantizedLRNProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 1U) << "Input:[data]";
    const TShape &dshape = in_shape->at(0);
    if (dshape.ndim() == 0) return false;
    out_shape->clear();
    out_shape->push_back(dshape);
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_GE(in_type->size(), 1U);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    for (index_t i = 0; i < in_type->size(); ++i) {
      if ((*in_type)[i] == -1) {
        (*in_type)[i] = dtype;
      } else {
        CHECK_EQ((*in_type)[i], dtype)
          << "This layer requires uniform type. "
          << "Expected " << dtype << " v.s. given "
          << (*in_type)[i] << " at " << ListArguments()[i];
      }
    }
    int n_out = this->ListOutputs().size();
    out_type->clear();
    for (int i = 0; i < n_out; ++i ) out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new QuantizedLRNProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "quantized_lrn";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[0], in_data[0], out_data[0]};
  }

  int NumVisibleOutputs() const override {
    return 1;
  }

  int NumOutputs() const override {
    return 1;
  }

  std::vector<std::string> ListArguments() const override {
    return {"data"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output"};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  QuantizedLRNParam param_;
};  // QuantizedLRNProp
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_QUANTIZED_LRN_INL_H_
