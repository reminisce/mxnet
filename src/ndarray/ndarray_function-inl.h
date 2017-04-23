/*!
 *  Copyright (c) 2015 by Contributors
 * \file ndarray_function-inl.h
 * \brief The real implementation of NDArray functions.
 */
#ifndef MXNET_NDARRAY_NDARRAY_FUNCTION_INL_H_
#define MXNET_NDARRAY_NDARRAY_FUNCTION_INL_H_

#include <vector>
#include <mxnet/ndarray.h>
#include "./ndarray_function.h"
// this file will be included twice by CPU and GPU
// macro to help specialize evaluation function

#ifndef DECL_TERNARY
#define DECL_TERNARY(XPU, OP, FUN)                                                     \
  template<>                                                                           \
  void Eval<XPU, OP>(const TBlob &lhs, const TBlob &mhs,                               \
                                       const TBlob &rhs, TBlob *ret, RunContext ctx) { \
    FUN<XPU, OP>(lhs, mhs, rhs, ret, ctx);                                             \
  }
#endif

#ifndef DECL_BINARY
#define DECL_BINARY(XPU, OP, FUN)                                                            \
  template<>                                                                                 \
  void Eval<XPU, OP>(const TBlob &lhs, const TBlob &rhs, TBlob *ret, RunContext ctx) {       \
    FUN<XPU, OP>(lhs, rhs, ret, ctx);                                                        \
  }
#endif

#ifndef DECL_BINARY_SPARSE
#define DECL_BINARY_SPARSE(XPU, OP, FUN)                                                     \
  template<>                                                                                 \
  void Eval<XPU, OP>(const NDArray& lhs, const NDArray& rhs, NDArray* ret, RunContext ctx) { \
    FUN<XPU, OP>(lhs, rhs, ret, ctx);                                                        \
  }
#endif

#ifndef DECL_SCALAR
#define DECL_SCALAR(XPU, OP, FUN, REVERSE)                              \
  template<>                                                            \
  void Eval<XPU, OP, REVERSE>(const TBlob &lhs, const real_t &rhs, TBlob *ret, RunContext ctx) { \
    FUN<XPU, OP, REVERSE>(lhs, rhs, ret, ctx);                          \
  }
#endif

#if defined(__CUDACC__)
#define DEVICE gpu
#else
#define DEVICE cpu
#endif

namespace mxnet {
namespace ndarray {

#define NDARRAY_IDX_TYPE_SWITCH(type, DType, ...)   \
  switch (type) {                                   \
  case mshadow::kUint8:                             \
    {                                               \
      typedef uint8_t DType;                        \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case mshadow::kInt32:                             \
    {                                               \
      typedef int32_t DType;                        \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  default:                                          \
    LOG(FATAL) << "Unknown idx type enum " << type; \
  }

// true implementation
template<typename xpu, typename OP>
inline void EvalBinary_(const TBlob &lhs, const TBlob &rhs,
                        TBlob *ret, RunContext ctx) {
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(ret->type_flag_, lhs.type_flag_)
    << "Only support input/output with the same data type";
  CHECK_EQ(ret->type_flag_, rhs.type_flag_)
    << "Only support input/output with the same data type";
  MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
    ret->FlatTo2D<xpu, DType>(s)
      = F<typename OP::mshadow_op>(lhs.FlatTo2D<xpu, DType>(s),
                                   rhs.FlatTo2D<xpu, DType>(s));
  });
}

template<typename xpu, typename OP>
inline void EvalBinaryCsrOpCsr(const NDArray& lhs, const NDArray& rhs,
                               NDArray* ret, RunContext ctx) {
  // TODO(junwu): use enum to get indptr, col idx, etc.
  NDARRAY_IDX_TYPE_SWITCH(lhs.aux_type(0), IType, {  // indptr data type
    NDARRAY_IDX_TYPE_SWITCH(lhs.aux_type(1), CType, {  // col idx data type
      MSHADOW_TYPE_SWITCH(lhs.data().type_flag_, DType, {  // ndarray data type
        const index_t num_rows = lhs.shape()[0];  // number of rows
        index_t left_nnz = lhs.data().Size();  // number of non-zeros of the left
        const IType* left_indptr = lhs.aux_data(0).dptr<IType>();  // len = num_rows + 1
        const CType* left_col_idx = lhs.aux_data(1).dptr<CType>();  // len = left_nnz
        const DType* left_values = lhs.data().dptr<DType>();  // len = left_nnz

        index_t right_nnz = rhs.data().Size();  // number of non-zeros of the right
        const IType* right_indptr = rhs.aux_data(0).dptr<IType>();  // len = num_rows + 1
        const CType* right_col_idx = rhs.aux_data(1).dptr<CType>();  // len = right_nnz
        const DType* right_values = rhs.data().dptr<DType>();  // len = right_nnz

        // TODO(junwu): verify this has been done correctly for csr
        ret->CheckAndAlloc({TShape({num_rows+1}), TShape({left_nnz+right_nnz})});
        // Note: ret_nnz <= left_nnz + right_nnz
        IType* ret_indptr = ret->aux_data(0).dptr<IType>();  // len = num_rows+1
        CType* ret_col_idx = ret->aux_data(1).dptr<CType>();  // len = ret_nnz
        DType* ret_values = ret->data().dptr<DType>();  // len = ret_nnz

        auto op_func = OP::mshadow_op::template Map<DType>;

        const DType dtype_zero = static_cast<DType>(0);
        ret_indptr[0] = 0;
        for (index_t i = 0; i < num_rows; ++i) {
          const auto left_nnz_row = left_indptr[i+1] - left_indptr[i];  // num of nnz in i-th row
          const auto right_nnz_row = right_indptr[i+1] - right_indptr[i];  // num of nnz i-th row
          if (left_nnz_row == 0 && right_nnz_row == 0) {  // no nnz elements in both rows
            ret_indptr[i+1] = ret_indptr[i];
          } else if (left_nnz_row == 0 || right_nnz_row == 0) {  // only one has nnz elements
            bool has_left_nnz = (0 == right_nnz_row);
            const IType* indptr = (has_left_nnz? right_indptr : left_indptr);
            const CType* col_idx = (has_left_nnz? right_col_idx : left_col_idx);
            const DType* values = (has_left_nnz? right_values : left_values);
            const auto nnz_row = (has_left_nnz? right_nnz_row : left_nnz_row);
            ret_indptr[i+1] = ret_indptr[i] + nnz_row;
            auto k = ret_indptr[i];
            for (auto j = indptr[i]; j < indptr[i+1]; ++j, ++k) {
              const DType op_res = (has_left_nnz? op_func(values[j], dtype_zero)
                                    : op_func(dtype_zero, values[j]));
              if (op_res == dtype_zero) continue;
              ret_values[k] = op_res;
              ret_col_idx[k] = col_idx[j];
            }
          } else {  // both have nnz elements
            IType ret_nnz_row = 0;
            auto k = ret_indptr[i];
            auto j1 = left_indptr[i], j2 = right_indptr[i];
            while (j1 < left_indptr[i+1] && j2 < right_indptr[i+1]) {
              const auto left_col = left_col_idx[j1];
              const auto right_col = right_col_idx[j2];
              if (left_col == right_col) {
                const DType op_res = op_func(left_values[j1], right_values[j2]);
                if (op_res == dtype_zero) continue;
                ret_col_idx[k] = left_col;
                ret_values[k] = op_res;
                ++j1; ++j2; ++k;
              } else if (left_col < right_col) {
                const DType op_res = op_func(left_values[j1], dtype_zero);
                if (op_res == dtype_zero) continue;
                ret_col_idx[k] = left_col;
                ret_values[k] = op_res;
                ++j1; ++k;
              } else {
                const DType op_res = op_func(dtype_zero, right_values[j2]);
                if (op_res == dtype_zero) continue;
                ret_col_idx[k] = right_col;
                ret_values[k] = op_res;
                ++j2; ++k;
              }
              ++ret_nnz_row;
            }
            while (j1 < left_indptr[i+1]) {
              const DType op_res = op_func(left_values[j1], dtype_zero);
              if (op_res == dtype_zero) continue;
              ret_col_idx[k] = left_col_idx[j1];
              ret_values[k] = op_res;
              ++j1; ++k;
              ++ret_nnz_row;
            }
            while (j2 < right_indptr[i+1]) {
              const DType op_res = op_func(dtype_zero, right_values[j2]);
              if (op_res == dtype_zero) continue;
              ret_col_idx[k] = right_col_idx[j2];
              ret_values[k] = op_res;
              ++j2; ++k;
              ++ret_nnz_row;
            }
            ret_indptr[i+1] = ret_indptr[i] + ret_nnz_row;
          }
        }

        // reset shapes of ret.data() and ret.aux_shape(1);
        // ret.indptr has the same size as lhs and rhs
        TShape ret_shape(1);
        ret_shape[0] = ret_indptr[num_rows];
        ret->SetStorageShape(ret_shape);
        // TODO(junwu): replace 1 with the column array enum
        ret->SetAuxShape(1, ret_shape);
      });
    });
  });
}

template<typename xpu, typename OP>
inline void EvalBinary_(const NDArray& lhs, const NDArray& rhs,
                        NDArray* ret, RunContext ctx) {
  if (lhs.storage_type() == kCSRStorage && rhs.storage_type() == kCSRStorage) {
    EvalBinaryCsrOpCsr<xpu, OP>(lhs, rhs, ret, ctx);
  } else {
    LOG(FATAL) << "Binary sparse op only implemented for csr type";
  }
}

template<typename xpu, typename OP>
inline void EvalOneHot_(const TBlob &index, const TBlob &rhs,
                        TBlob *ret, RunContext ctx) {
  LOG(INFO) << "The operator onehot_encode is deprecated; use one_hot instead.";
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  // TODO(eric): support mixed type encoding, i.e. int index and float rhs.
  CHECK_EQ(ret->type_flag_, mshadow::default_type_flag)
    << "one_hot_encode only support float32 as input/output";
  CHECK_EQ(rhs.type_flag_, mshadow::default_type_flag)
    << "one_hot_encode only support float32 as input/output";
  CHECK_EQ(index.type_flag_, mshadow::default_type_flag)
    << "one_hot_encode only support float32 as input/output";
  ret->get<xpu, 2, real_t>(s) =
    one_hot_encode(index.get<xpu, 1, real_t>(s),
                   rhs.shape_[1]);
}

// TODO(junwu): implement this function
template<typename xpu, typename OP>
inline void EvalOneHot_(const NDArray& index, const NDArray& rhs,
                        NDArray* ret, RunContext ctx) {
  LOG(FATAL) << "OneHot not implemented for sparse ndarrays";
}

template<typename xpu, typename OP>
inline void EvalMatChooseRowElem_(const TBlob &lhs, const TBlob &rhs,
                                  TBlob *ret, RunContext ctx) {
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  // TODO(eric): support mixed type choose, i.e. int index and float rhs.
  CHECK_EQ(ret->type_flag_, mshadow::default_type_flag)
    << "mat_choose_row_element only support float32 as input/output";
  CHECK_EQ(rhs.type_flag_, mshadow::default_type_flag)
    << "mat_choose_row_element only support float32 as input/output";
  CHECK_EQ(lhs.type_flag_, mshadow::default_type_flag)
    << "mat_choose_row_element only support float32 as input/output";
  ret->get<xpu, 1, real_t>(s)
      = mat_choose_row_element(lhs.get<xpu, 2, real_t>(s),
                               rhs.get<xpu, 1, real_t>(s));
}

// TODO(junwu): implement this function
template<typename xpu, typename OP>
inline void EvalMatChooseRowElem_(const NDArray& lhs, const NDArray& rhs,
                                  NDArray* ret, RunContext ctx) {
  LOG(FATAL) << "ChooseRowELem not implemented for sparse ndarrays";
}

template<typename xpu, typename OP>
inline void EvalMatFillRowElem_(const TBlob &lhs, const TBlob &mhs, const TBlob &rhs,
                                  TBlob *ret, RunContext ctx) {
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  ret->get<xpu, 2, real_t>(s)
          = mat_fill_row_element(lhs.get<xpu, 2, real_t>(s),
                                 mhs.get<xpu, 1, real_t>(s),
                                 rhs.get<xpu, 1, real_t>(s));
}

template<typename xpu, typename OP, bool reverse>
inline void EvalScalar_(const TBlob &lhs, const real_t &rhs,
                        TBlob *ret, RunContext ctx) {
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(ret->type_flag_, lhs.type_flag_)
    << "Only support input/output with the same data type";
  if (reverse) {
    MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
      ret->FlatTo2D<xpu, DType>(s)
        = F<typename OP::mshadow_op>(scalar(DType(rhs)), lhs.FlatTo2D<xpu, DType>(s));
    });
  } else {
    MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
      ret->FlatTo2D<xpu, DType>(s)
        = F<typename OP::mshadow_op>(lhs.FlatTo2D<xpu, DType>(s), scalar(DType(rhs)));
    });
  }
}

template<typename xpu, typename OP, bool reverse>
inline void EvalScalar_(const NDArray& lhs, const real_t &rhs,
                        NDArray* ret, RunContext ctx) {
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(ret->data().type_flag_, lhs.data().type_flag_)
    << "Only support input/output with the same data type";
  // TODO(junwu): use enum to get indptr, col idx, etc.
  MSHADOW_TYPE_SWITCH(lhs.aux_type(0), IType, {  // indptr data type
    MSHADOW_TYPE_SWITCH(lhs.aux_type(1), CType, {  // col idx data type
      MSHADOW_TYPE_SWITCH(lhs.data().type_flag_, DType, {  // ndarray data type
      });
    });
  });
}

template<>
void EvalClip<DEVICE>(const TBlob &src, const real_t &a_min, const real_t &a_max,
                      TBlob *ret, RunContext ctx) {
  typedef DEVICE xpu;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(ret->type_flag_, src.type_flag_)
    << "Only support input/output with the same data type";
  MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
    ret->FlatTo2D<xpu, DType>(s)
      = F<ClipMax::mshadow_op>(
          F<ClipMin::mshadow_op>(src.FlatTo2D<xpu, DType>(s), scalar(DType(a_min))),
          scalar(DType(a_max)));
  });
}

template<>
void EvalRandom<DEVICE, UniformDistribution>(
    const real_t &a,
    const real_t &b,
    const Resource &resource,
    TBlob *ret,
    RunContext ctx) {
  typedef DEVICE xpu;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  switch (ret->type_flag_) {
  case mshadow::kFloat32:
    {
      mshadow::Random<xpu, float> *prnd = resource.get_random<xpu, float>(s);
      mshadow::Tensor<xpu, 2, float> tmp = ret->FlatTo2D<xpu, float>(s);
      prnd->SampleUniform(&tmp, float(a), float(b));  // NOLINT(*)
      break;
    }
  case mshadow::kFloat64:
    {
      mshadow::Random<xpu, double> *prnd = resource.get_random<xpu, double>(s);
      mshadow::Tensor<xpu, 2, double> tmp = ret->FlatTo2D<xpu, double>(s);
      prnd->SampleUniform(&tmp, double(a), double(b));  // NOLINT(*)
      break;
    }
  default:
    LOG(FATAL) << "Random only support float32 and float64";
  }
}

template<>
void EvalRandom<DEVICE, GaussianDistribution>(
    const real_t &mu,
    const real_t &sigma,
    const Resource &resource,
    TBlob *ret,
    RunContext ctx) {
  typedef DEVICE xpu;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  switch (ret->type_flag_) {
  case mshadow::kFloat32:
    {
      mshadow::Random<xpu, float> *prnd = resource.get_random<xpu, float>(s);
      mshadow::Tensor<xpu, 2, float> tmp = ret->FlatTo2D<xpu, float>(s);
      prnd->SampleGaussian(&tmp, float(mu), float(sigma));  // NOLINT(*)
      break;
    }
  case mshadow::kFloat64:
    {
      mshadow::Random<xpu, double> *prnd = resource.get_random<xpu, double>(s);
      mshadow::Tensor<xpu, 2, double> tmp = ret->FlatTo2D<xpu, double>(s);
      prnd->SampleGaussian(&tmp, double(mu), double(sigma));  // NOLINT(*)
      break;
    }
  default:
    LOG(FATAL) << "Random only support float32 and float64";
  }
}

template<>
void EvalRandom<DEVICE, GammaDistribution>(
    const real_t &alpha,
    const real_t &beta,
    const Resource &resource,
    TBlob *ret,
    RunContext ctx) {
  typedef cpu xpu;  // No support for gpu for this distribution.
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  switch (ret->type_flag_) {
  case mshadow::kFloat32:
    {
      mshadow::Random<xpu, float> *prnd = resource.get_random<xpu, float>(s);
      mshadow::Tensor<xpu, 2, float> tmp = ret->FlatTo2D<xpu, float>(s);
      prnd->SampleGamma(&tmp, float(alpha), float(beta));  // NOLINT(*)
      break;
    }
  case mshadow::kFloat64:
    {
      mshadow::Random<xpu, double> *prnd = resource.get_random<xpu, double>(s);
      mshadow::Tensor<xpu, 2, double> tmp = ret->FlatTo2D<xpu, double>(s);
      prnd->SampleGamma(&tmp, double(alpha), double(beta));  // NOLINT(*)
      break;
    }
  default:
    LOG(FATAL) << "Random only support float32 and float64";
  }
}


template<>
void EvalRandom<DEVICE, ExponentialDistribution>(
    const real_t &lambda,
    const real_t &dummy,  // this is to satisfy the SampleOp lambda signature
    const Resource &resource,
    TBlob *ret,
    RunContext ctx) {
  typedef cpu xpu;  // No support for gpu for this distribution.
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  switch (ret->type_flag_) {
  case mshadow::kFloat32:
    {
      mshadow::Random<xpu, float> *prnd = resource.get_random<xpu, float>(s);
      mshadow::Tensor<xpu, 2, float> tmp = ret->FlatTo2D<xpu, float>(s);
      prnd->SampleExponential(&tmp, float(lambda));  // NOLINT(*)
      break;
    }
  case mshadow::kFloat64:
    {
      mshadow::Random<xpu, double> *prnd = resource.get_random<xpu, double>(s);
      mshadow::Tensor<xpu, 2, double> tmp = ret->FlatTo2D<xpu, double>(s);
      prnd->SampleExponential(&tmp, double(lambda));  // NOLINT(*)
      break;
    }
  default:
    LOG(FATAL) << "Random only support float32 and float64";
  }
}

template<>
void EvalRandom<DEVICE, PoissonDistribution>(
    const real_t &lambda,
    const real_t &dummy,  // this is to satisfy the SampleOp lambda signature
    const Resource &resource,
    TBlob *ret,
    RunContext ctx) {
  typedef cpu xpu;  // No support for gpu for this distribution.
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  switch (ret->type_flag_) {
  case mshadow::kFloat32:
    {
      mshadow::Random<xpu, float> *prnd = resource.get_random<xpu, float>(s);
      mshadow::Tensor<xpu, 2, float> tmp = ret->FlatTo2D<xpu, float>(s);
      prnd->SamplePoisson(&tmp, float(lambda));  // NOLINT(*)
      break;
    }
  case mshadow::kFloat64:
    {
      mshadow::Random<xpu, double> *prnd = resource.get_random<xpu, double>(s);
      mshadow::Tensor<xpu, 2, double> tmp = ret->FlatTo2D<xpu, double>(s);
      prnd->SamplePoisson(&tmp, double(lambda));  // NOLINT(*)
      break;
    }
  default:
    LOG(FATAL) << "Random only support float32 and float64";
  }
}

template<>
void EvalRandom<DEVICE, NegBinomialDistribution>(
    const real_t &k,
    const real_t &p,
    const Resource &resource,
    TBlob *ret,
    RunContext ctx) {
  typedef cpu xpu;  // No support for gpu for this distribution.
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  switch (ret->type_flag_) {
  case mshadow::kFloat32:
    {
      mshadow::Random<xpu, float> *prnd = resource.get_random<xpu, float>(s);
      mshadow::Tensor<xpu, 2, float> tmp = ret->FlatTo2D<xpu, float>(s);
      prnd->SampleNegativeBinomial(&tmp, float(k), float(p));  // NOLINT(*)
      break;
    }
  case mshadow::kFloat64:
    {
      mshadow::Random<xpu, double> *prnd = resource.get_random<xpu, double>(s);
      mshadow::Tensor<xpu, 2, double> tmp = ret->FlatTo2D<xpu, double>(s);
      prnd->SampleNegativeBinomial(&tmp, double(k), double(p));  // NOLINT(*)
      break;
    }
  default:
    LOG(FATAL) << "Random only support float32 and float64";
  }
}

template<>
void EvalRandom<DEVICE, GenNegBinomialDistribution>(
    const real_t &mu,
    const real_t &alpha,
    const Resource &resource,
    TBlob *ret,
    RunContext ctx) {
  typedef cpu xpu;  // No support for gpu for this distribution.
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  switch (ret->type_flag_) {
  case mshadow::kFloat32:
    {
      mshadow::Random<xpu, float> *prnd = resource.get_random<xpu, float>(s);
      mshadow::Tensor<xpu, 2, float> tmp = ret->FlatTo2D<xpu, float>(s);
      prnd->SampleGeneralizedNegativeBinomial(&tmp, float(mu), float(alpha));  // NOLINT(*)
      break;
    }
  case mshadow::kFloat64:
    {
      mshadow::Random<xpu, double> *prnd = resource.get_random<xpu, double>(s);
      mshadow::Tensor<xpu, 2, double> tmp = ret->FlatTo2D<xpu, double>(s);
      prnd->SampleGeneralizedNegativeBinomial(&tmp, double(mu), double(alpha));  // NOLINT(*)
      break;
    }
  default:
    LOG(FATAL) << "Random only support float32 and float64";
  }
}

template<>
void Eval<DEVICE>(const real_t &rhs, TBlob *ret, RunContext ctx) {
  mshadow::Stream<DEVICE> *s = ctx.get_stream<DEVICE>();
  MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
    ret->FlatTo2D<DEVICE, DType>(s) = DType(rhs);
  });
}

template<>
void ElementwiseSum<DEVICE>(const std::vector<TBlob> source,
                            TBlob *dst,
                            RunContext ctx) {
  typedef DEVICE xpu;
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  for (size_t i = 1; i < source.size(); ++i) {
    CHECK_EQ(source[i].type_flag_, dst->type_flag_)
      << "Only support input/output with the same data type";
  }
  MSHADOW_TYPE_SWITCH(dst->type_flag_, DType, {
    Tensor<xpu, 2, DType> out = dst->FlatTo2D<xpu, DType>(s);

    switch (source.size()) {
      case 2: {
        Tensor<xpu, 2, DType> in_0 = source[0].FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> in_1 = source[1].FlatTo2D<xpu, DType>(s);
        out = in_0 + in_1;
        break;
      }
      case 3: {
        Tensor<xpu, 2, DType> in_0 = source[0].FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> in_1 = source[1].FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> in_2 = source[2].FlatTo2D<xpu, DType>(s);
        out = in_0 + in_1 + in_2;
        break;
      }
      case 4: {
        Tensor<xpu, 2, DType> in_0 = source[0].FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> in_1 = source[1].FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> in_2 = source[2].FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> in_3 = source[3].FlatTo2D<xpu, DType>(s);
        out = in_0 + in_1 + in_2 + in_3;
        break;
      }
      default: {
        Tensor<xpu, 2, DType> in_0 = source[0].FlatTo2D<xpu, DType>(s);
        out = F<mshadow::op::identity>(in_0);
        for (size_t i = 1; i < source.size(); ++i) {
          out += source[i].FlatTo2D<xpu, DType>(s);
        }
        break;
      }
    }
  });
}

template <>
void EvalBroadcast<DEVICE>(TBlob const& src, TBlob* ret, int size, RunContext ctx) {
  typedef DEVICE xpu;
  mshadow::Stream<xpu>* s = ctx.get_stream<xpu>();
  mshadow::Tensor<xpu, 3> out = ret->get<xpu, 3, real_t>(s);
  mshadow::Tensor<xpu, 2> in = src.get<xpu, 2, real_t>(s);
  out = mshadow::expr::broadcast_with_axis(in, 0, size);
}

// declarations
DECL_BINARY(DEVICE, MatChooseRowElem, EvalMatChooseRowElem_)
DECL_TERNARY(DEVICE, MatFillRowElem, EvalMatFillRowElem_)
DECL_BINARY(DEVICE, OneHotEncode, EvalOneHot_)
DECL_BINARY(DEVICE, Plus, EvalBinary_)
DECL_BINARY(DEVICE, Minus, EvalBinary_)
DECL_BINARY(DEVICE, Mul, EvalBinary_)
DECL_BINARY(DEVICE, Div, EvalBinary_)
DECL_SCALAR(DEVICE, Plus, EvalScalar_, true)
DECL_SCALAR(DEVICE, Minus, EvalScalar_, true)
DECL_SCALAR(DEVICE, Mul, EvalScalar_, true)
DECL_SCALAR(DEVICE, Div, EvalScalar_, true)

DECL_BINARY_SPARSE(DEVICE, MatChooseRowElem, EvalMatChooseRowElem_)
DECL_BINARY_SPARSE(DEVICE, OneHotEncode, EvalOneHot_)
DECL_BINARY_SPARSE(DEVICE, Plus, EvalBinary_)
DECL_BINARY_SPARSE(DEVICE, Minus, EvalBinary_)
DECL_BINARY_SPARSE(DEVICE, Mul, EvalBinary_)
DECL_BINARY_SPARSE(DEVICE, Div, EvalBinary_)

// for reverse seq
DECL_SCALAR(DEVICE, Plus, EvalScalar_, false)
DECL_SCALAR(DEVICE, Minus, EvalScalar_, false)
DECL_SCALAR(DEVICE, Mul, EvalScalar_, false)
DECL_SCALAR(DEVICE, Div, EvalScalar_, false)
}  // namespace ndarray
}  // namespace mxnet

#endif  // MXNET_NDARRAY_NDARRAY_FUNCTION_INL_H_
