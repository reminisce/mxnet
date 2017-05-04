/*!
 * Copyright (c) 2017 by Contributors
 * \file quantized_bias_add_op-inl.h
 * \brief quantized bias_add operator and symbol
 * \author Ziheng Jiang
*/
#ifndef MXNET_OPERATOR_QUANTIZED_BIAS_ADD_INL_H_
#define MXNET_OPERATOR_QUANTIZED_BIAS_ADD_INL_H_

#include "../operator_common.h"
#include "../elemwise_op_common.h"


namespace mxnet {
namespace op {

struct QuantizedBiasAddParam :
  public dmlc::Parameter<QuantizedBiasAddParam> {
  DMLC_DECLARE_PARAMETER(QuantizedBiasAddParam) {
  }
};

template<typename xpu>
Operator* CreateOp(int dtype,
                   const Context& ctx,
                   const std::vector<TShape>& in_shape,
                   const std::vector<TShape>& out_shape,
                   const QuantizedBiasAddParam& param);

class QuantizedBiasAddProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"data", "bias", "min_data", "max_data", "min_bias", "max_bias"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "min_out", "max_out"};
  }

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
    CHECK_EQ(in_shape->size(), 6U);

    dshape = in_shape->at(0);
    CHECK(!shape_is_none(dshape));
    TShape bshape{dshape[-1]};
    SHAPE_ASSIGN_CHECK(*in_shape, 1, bshape);

    for (size_t i = 2; i < 6; ++i) {
      SHAPE_ASSIGN_CHECK(*in_shape, i, bshape);
    }

    out_shape->clear();
    out_shape->push_back(dshape);
    out_shape->push_back(TShape{1});
    out_shape->push_back(TShape{1});
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_EQ(in_type->size(), 6U);
    CHECK_EQ((*in_type)[0], mshadow::kInt8);

    for (size_t i = 2; i < 6; ++i) {
      TYPE_ASSIGN_CHECK(*in_type, i, mshadow::kFloat32);
    }

    out_type->clear();
    out_type->push_back(mshadow::kInt8);
    out_type->push_back(mshadow::kFloat32);
    out_type->push_back(mshadow::kFloat32);
    return true;
  }

  OperatorProperty* Copy() const override {
    QuantizedBiasAddProp* prop = new QuantizedBiasAddProp();
    prop->param_ = this->param_;
    return prop;
  }

  std::string TypeString() const override {
    return "quantized_bias_add";
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  QuantizedBiasAddParam param_;
};  // class QuantizedBiasAddSymbol

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_QUANTIZED_BIAS_ADD_INL_H_
