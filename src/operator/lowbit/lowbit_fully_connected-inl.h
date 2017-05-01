/*!
 * Copyright (c) 2015 by Contributors
 * \file lowbit_fully_connect_op-inl.h
 * \brief lowbit fully connect operator and symbol
 * \author Ziheng Jiang
*/
#ifndef MXNET_OPERATOR_LOWBIT_FULLY_CONNECTED_INL_H_
#define MXNET_OPERATOR_LOWBIT_FULLY_CONNECTED_INL_H_

#include "../operator_common.h"
#include "../elemwise_op_common.h"


namespace mxnet {
namespace op {

struct LowbitFullyConnectedParam :
  public dmlc::Parameter<LowbitFullyConnectedParam> {
  int num_hidden;
  DMLC_DECLARE_PARAMETER(LowbitFullyConnectedParam) {
    DMLC_DECLARE_FIELD(num_hidden).set_lower_bound(1)
    .describe("Number of hidden nodes of the output.");
  }
};

template<typename xpu>
Operator* CreateOp(int dtype,
                   const Context& ctx,
                   const std::vector<TShape>& in_shape,
                   const std::vector<TShape>& out_shape,
                   const LowbitFullyConnectedParam& param);

#if DMLC_USE_CXX11
class LowbitFullyConnectedProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"data", "weight"};
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  // (batch_size, input_dim) * (num_hidden, input_dim) = (batch_size, num_hidden)
  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 2U) << "Input:[data, weight]";
    CHECK_EQ(out_shape->size(), 1U);
    TShape dshape = (*in_shape)[0];
    TShape oshape = (*out_shape)[0];
    // require data to be known
    if (dshape.ndim() ==  0) return false;

    index_t num_input = dshape.ProdShape(1, dshape.ndim());
    SHAPE_ASSIGN_CHECK(*in_shape, 1, Shape2(param_.num_hidden, num_input));
    SHAPE_ASSIGN_CHECK(*out_shape, 0, Shape2(dshape[0], param_.num_hidden));
    if (oshape.ndim() != 0) {
      dshape[0] = oshape[0];
      SHAPE_ASSIGN_CHECK(*in_shape, 0, dshape);
    }
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_GE(in_type->size(), 1U);
    nnvm::NodeAttrs attrs;
    attrs.name = "lowbit_fully_connected";
    return ElemwiseType<2, 1>(attrs, in_type, out_type);
  }

  OperatorProperty* Copy() const override {
    LowbitFullyConnectedProp* prop = new LowbitFullyConnectedProp();
    prop->param_ = this->param_;
    return prop;
  }

  std::string TypeString() const override {
    return "lowbit_fully_connected";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[0], in_data[0], in_data[0]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{in_data[0], in_grad[0]}};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  LowbitFullyConnectedParam param_;
};  // class LowbitFullyConnectedSymbol
#endif
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_lowbit_FULLY_CONNECTED_INL_H_
