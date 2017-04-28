/*!
 * Copyright (c) 2017 by Contributors
 * \file quantized_max_pool-inl.h
 * \brief
 * \author Ziheng Jiang
*/

#ifndef MXNET_OPERATOR_CONTRIB_QUANTIZED_MAX_POOL_INL_H_
#define MXNET_OPERATOR_CONTRIB_QUANTIZED_MAX_POOL_INL_H_

#include "../operator_common.h"
#include "../nn/pool.h"

namespace mxnet {
namespace op {

struct QuantizedMaxPoolParam : public dmlc::Parameter<QuantizedMaxPoolParam> {
  TShape kernel;
  TShape stride;
  TShape pad;
  int convention;
  bool global_pool;
  DMLC_DECLARE_PARAMETER(QuantizedMaxPoolParam) {
    DMLC_DECLARE_FIELD(global_pool).set_default(false)
    .describe("Ignore kernel size, do global pooling based on current input feature map. ");

    DMLC_DECLARE_FIELD(kernel)
    .enforce_nonzero()
    .describe("pooling kernel size: (y, x) or (d, y, x)");

    DMLC_DECLARE_FIELD(convention).set_default(pool_enum::kValid)
    .add_enum("full", pool_enum::kFull)
    .add_enum("valid", pool_enum::kValid)
    .describe("Pooling convention to be applied.");

    DMLC_DECLARE_FIELD(stride).set_default(TShape())
    .enforce_nonzero()
    .describe("stride: for pooling (y, x) or (d, y, x)");

    DMLC_DECLARE_FIELD(pad).set_default(TShape())
    .describe("pad for pooling: (y, x) or (d, y, x)");
  }
};

template<typename xpu>
Operator* CreateOp(QuantizedMaxPoolParam param, int dtype);


class QuantizedMaxPoolProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    using namespace mshadow;
    param_.Init(kwargs);
    if (param_.kernel.ndim() == 1) {
      if (param_.stride.ndim() == 0) param_.stride = Shape1(1);
      if (param_.pad.ndim() == 0) param_.pad = Shape1(0);
    } else if (param_.kernel.ndim() == 2) {
      if (param_.stride.ndim() == 0) param_.stride = Shape2(1, 1);
      if (param_.pad.ndim() == 0) param_.pad = Shape2(0, 0);
    } else {
      CHECK_EQ(param_.kernel.ndim(), 3U) << param_.kernel.ndim() << "D pooling not supported";
      if (param_.stride.ndim() == 0) param_.stride = Shape3(1, 1, 1);
      if (param_.pad.ndim() == 0) param_.pad = Shape3(0, 0, 0);
    }
    CHECK_EQ(param_.stride.ndim(), param_.kernel.ndim())
      << "stride and kernel should have the same length";
    CHECK_EQ(param_.pad.ndim(), param_.kernel.ndim())
      << "pad and kernel should have the same length";
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    CHECK_EQ(in_shape->size(), 1U);
    const TShape &dshape = (*in_shape)[0];
    CHECK_GE(dshape.ndim(), 3U) << "Pooling: Input data should be  3D in (batch, channel, x)"
                                << " Or 4D in (batch, channel, y, x) "
                                << " Or 5D in (batch, channel, d, y, x)";
    TShape oshape = dshape;
    if (dshape.ndim() ==  0) return false;
    if (param_.kernel.ndim() == 1) {
      CHECK_EQ(dshape.ndim(), 3U) << "Pooling: Input data should be 3D in (batch, channel, x)";
      if (param_.global_pool) {
        oshape[2] = 1;
      } else {
        CHECK(param_.kernel[0] <= dshape[2] + 2 * param_.pad[0])
            << "kernel size (" << param_.kernel[0] << ") exceeds input (" << dshape[2]
            << " padded to " << (dshape[2] + 2*param_.pad[0]) << ")";
        if (param_.convention == pool_enum::kValid) {
          oshape[2] = 1 + (dshape[2] + 2 * param_.pad[0] - param_.kernel[0]) /
                              param_.stride[0];
        } else {
          oshape[2] = 1 + static_cast<int>(ceil(static_cast<float>(
                              dshape[2] + 2 * param_.pad[0] -
                              param_.kernel[0]) / param_.stride[0]));
        }
      }
      out_shape->clear();
      out_shape->push_back(oshape);  // save output shape
    } else if (param_.kernel.ndim() == 2) {
      CHECK_EQ(dshape.ndim(), 4U) << "Pooling: Input data should be 4D in (batch, channel, y, x)";
      if (param_.global_pool) {
        oshape[2] = 1;
        oshape[3] = 1;
      } else {
        CHECK(param_.kernel[0] <= dshape[2] + 2 * param_.pad[0])
            << "kernel size (" << param_.kernel[0] << ") exceeds input (" << dshape[2]
            << " padded to " << (dshape[2] + 2*param_.pad[0]) << ")";
        CHECK(param_.kernel[1] <= dshape[3] + 2 * param_.pad[1])
            << "kernel size (" << param_.kernel[1] << ") exceeds input (" << dshape[3]
            << " padded to " << (dshape[3] + 2*param_.pad[1]) << ")";
        if (param_.convention == pool_enum::kValid) {
          oshape[2] = 1 + (dshape[2] + 2 * param_.pad[0] - param_.kernel[0]) /
                              param_.stride[0];
          oshape[3] = 1 + (dshape[3] + 2 * param_.pad[1] - param_.kernel[1]) /
                              param_.stride[1];
        } else {
          oshape[2] = 1 + static_cast<int>(ceil(static_cast<float>(
                              dshape[2] + 2 * param_.pad[0] -
                              param_.kernel[0]) / param_.stride[0]));
          oshape[3] = 1 + static_cast<int>(ceil(static_cast<float>(
                              dshape[3] + 2 * param_.pad[1] -
                              param_.kernel[1]) / param_.stride[1]));
        }
      }
      out_shape->clear();
      out_shape->push_back(oshape);  // save output shape
    } else if (param_.kernel.ndim() == 3) {
      CHECK_EQ(dshape.ndim(), 5U)
        << "Pooling: Input data should be 5D in (batch, channel, d, y, x)";
      CHECK_LE(param_.kernel[0], dshape[2] + 2 * param_.pad[0]) << "kernel size exceeds input";
      CHECK_LE(param_.kernel[1], dshape[3] + 2 * param_.pad[1]) << "kernel size exceeds input";
      CHECK_LE(param_.kernel[2], dshape[4] + 2 * param_.pad[2]) << "kernel size exceeds input";
      if (param_.global_pool) {
        oshape[2] = 1;
        oshape[3] = 1;
        oshape[4] = 1;
      } else {
        if (param_.convention == pool_enum::kValid) {
          oshape[2] = 1 + (dshape[2] + 2 * param_.pad[0] - param_.kernel[0]) /
                              param_.stride[0];
          oshape[3] = 1 + (dshape[3] + 2 * param_.pad[1] - param_.kernel[1]) /
                              param_.stride[1];
          oshape[4] = 1 + (dshape[4] + 2 * param_.pad[2] - param_.kernel[2]) /
                              param_.stride[2];
        } else {
          oshape[2] = 1 + static_cast<int>(ceil(static_cast<float>(
                              dshape[2] + 2 * param_.pad[0] -
                              param_.kernel[0]) / param_.stride[0]));
          oshape[3] = 1 + static_cast<int>(ceil(static_cast<float>(
                              dshape[3] + 2 * param_.pad[1] -
                              param_.kernel[1]) / param_.stride[1]));
          oshape[4] = 1 + static_cast<int>(ceil(static_cast<float>(
                              dshape[4] + 2 * param_.pad[2] -
                              param_.kernel[2]) / param_.stride[2]));
        }
      }

      out_shape->clear();
      out_shape->push_back(oshape);  // save output shape
    }
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_EQ(in_type->size(), 1U);
    int dtype = (*in_type)[0];

    if (dtype == -1) {
      LOG(FATAL) << "Input type to pooling is not specified.";
      return false;
    }

    out_type->clear();
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    QuantizedMaxPoolProp *prop_sym = new QuantizedMaxPoolProp();
    prop_sym->param_ = this->param_;
    return prop_sym;
  }

  std::string TypeString() const override {
    return "quantized_max_pool";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[pool_enum::kOut], in_data[pool_enum::kData],
            out_data[pool_enum::kOut]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  QuantizedMaxPoolParam param_;
};  // class PoolingProp

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_POOLING_INL_H_
