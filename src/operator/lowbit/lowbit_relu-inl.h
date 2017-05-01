/*!
 * Copyright (c) 2017 by Contributors
 * \file lowbit_relu-inl.h
 * \brief
 * \author Ziheng Jiang
*/

#ifndef MXNET_OPERATOR_LOWBIT_RELU_INL_H_
#define MXNET_OPERATOR_LOWBIT_RELU_INL_H_
#include <mxnet/operator.h>
#include "../operator_common.h"

namespace mxnet {
namespace op {

template<typename xpu>
Operator* CreateLowbitReluOp(int dtype);

class LowbitReluProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {}

  std::map<std::string, std::string> GetParams() const override {
    return std::map<std::string, std::string>();
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
    out_type->clear();
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new LowbitReluProp();
    return ptr;
  }

  std::string TypeString() const override {
    return "lowbit_relu";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[0], out_data[0], in_data[0]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_grad[0], in_grad[0]}};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[0], out_data[0]}};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;
};


}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_lowbit_RELU_H_
