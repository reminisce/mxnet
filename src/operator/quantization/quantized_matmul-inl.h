/*!
 * Copyright (c) 2017 by Contributors
 * \file quantized_matmul_op-inl.h
 * \brief quantized matmul operator and symbol
 * \author Ziheng Jiang
*/
#ifndef MXNET_OPERATOR_QUANTIZED_MATMUL_INL_H_
#define MXNET_OPERATOR_QUANTIZED_MATMUL_INL_H_

#include "../operator_common.h"
#include "../elemwise_op_common.h"


namespace mxnet {
namespace op {

struct QuantizedMatmulParam :
  public dmlc::Parameter<QuantizedMatmulParam> {
  DMLC_DECLARE_PARAMETER(QuantizedMatmulParam) {
  }
};

template<typename xpu>
Operator* CreateOp(int dtype,
                   const Context& ctx,
                   const std::vector<TShape>& in_shape,
                   const std::vector<TShape>& out_shape,
                   const QuantizedMatmulParam& param);

class QuantizedMatmulProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"a", "b", "min_a", "max_a", "min_b", "max_b"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"out", "min_out", "max_out"};
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

    for (int i = 0; i < 2; ++i) {
      CHECK(!shape_is_none(in_shape->at(0)));
    }
    for (int i = 2; i < 6; ++i) {
      CHECK(shape_is_scalar(in_shape->at(i)));
    }
    TShape ashape = (*in_shape)[0];
    TShape bshape = (*in_shape)[1];
    CHECK_EQ(ashape[1], bshape[0]);

    out_shape->clear();
    out_shape->push_back(TShape{ashape[0], bshape[1]});
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
      CHECK_EQ((*in_type)[i], mshadow::kFloat32)
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
    QuantizedMatmulProp* prop = new QuantizedMatmulProp();
    prop->param_ = this->param_;
    return prop;
  }

  std::string TypeString() const override {
    return "quantized_matmul";
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  QuantizedMatmulParam param_;
};  // class QuantizedMatmulSymbol

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_QUANTIZED_MATMUL_INL_H_
