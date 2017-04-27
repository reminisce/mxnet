/*!
 *  Copyright (c) 2015 by Contributors
 * \file elementwise_unary_op-inl.h
 * \brief Function defintion of elementwise unary operators
 */
#ifndef MXNET_OPERATOR_TENSOR_ELEMWISE_UNARY_OP_H_
#define MXNET_OPERATOR_TENSOR_ELEMWISE_UNARY_OP_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <utility>
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"
#include "../special_functions-inl.h"
#include "../mxnet_op.h"

namespace mxnet {
namespace op {
template<typename GRAD_OP>
struct unary_bwd {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return DType(a*GRAD_OP::Map(b));
  }
};

template<typename xpu, typename OP>
void UnaryCompute(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Tensor<xpu, 1, DType> out = outputs[0].FlatTo1D<xpu, DType>(s);
    ASSIGN_DISPATCH(out, req[0], F<OP>(inputs[0].FlatTo1D<xpu, DType>(s)));
  });
}


template<typename xpu>
void IdentityCompute(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  if (req[0] == kNullOp) return;
  if (req[0] == kWriteInplace) {
    CHECK_EQ(inputs[0].dptr_, outputs[0].dptr_); return;
  }
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Tensor<xpu, 1, DType> out = outputs[0].FlatTo1D<xpu, DType>(s);
    ASSIGN_DISPATCH(out, req[0], F<mshadow_op::identity>(inputs[0].FlatTo1D<xpu, DType>(s)));
  });
}

template<typename xpu>
void IdentityComputeRsp(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<NDArray>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<NDArray>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  NDArrayStorageType storage_type = inputs[1].storage_type();
  CHECK_EQ(storage_type, kRowSparseStorage);
  if (req[0] == kNullOp) {
    LOG(FATAL) << "kNullOp in IdentityComputeEx not supported yet";
  }
  if (req[0] == kWriteInplace) {
    LOG(FATAL) << "kWriteInplace for sparse storage not supported yet";
  }
  TShape shape = inputs[1].aux_shape(rowsparse::kIdx);
  if (shape.ndim() == 0) return;
  outputs[0].CheckAndAlloc({shape});
  MSHADOW_TYPE_SWITCH(outputs[0].dtype(), DType, {
    MSHADOW_TYPE_SWITCH(outputs[0].aux_type(rowsparse::kIdx), AuxType, {
      auto out_d = outputs[0].data().FlatTo1D<xpu, DType>(s);
      auto out_aux = outputs[0].aux_data(rowsparse::kIdx).FlatTo1D<xpu, AuxType>(s);
      auto in_aux = inputs[1].aux_data(rowsparse::kIdx).FlatTo1D<xpu, AuxType>(s);
      ASSIGN_DISPATCH(out_d, req[0],
                      F<mshadow_op::identity>(inputs[1].data().FlatTo1D<xpu, DType>(s)));
      ASSIGN_DISPATCH(out_aux, req[0], F<mshadow_op::identity>(in_aux));
    });
  });
}

// FIXME the index is hard coded for _identity_with_attr_like_rhs op
template<typename xpu>
void IdentityComputeEx(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<NDArray>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<NDArray>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  CHECK_EQ(inputs.size(), 2);
  CHECK_EQ(outputs.size(), 1);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  NDArrayStorageType stype = inputs[1].storage_type();
  CHECK_EQ(stype, kRowSparseStorage) << "Not implemented yet";
  IdentityComputeRsp<xpu>(attrs, ctx, inputs, req, outputs);
}

struct CastParam : public dmlc::Parameter<CastParam> {
  // use int for enumeration
  int dtype;
  DMLC_DECLARE_PARAMETER(CastParam) {
    DMLC_DECLARE_FIELD(dtype)
    .add_enum("float32", mshadow::kFloat32)
    .add_enum("float64", mshadow::kFloat64)
    .add_enum("float16", mshadow::kFloat16)
    .add_enum("uint8", mshadow::kUint8)
    .add_enum("int32", mshadow::kInt32)
    .describe("Output data type.");
  }
};

inline bool CastType(const nnvm::NodeAttrs& attrs,
                     std::vector<int> *in_attrs,
                     std::vector<int> *out_attrs) {
  const CastParam& param = nnvm::get<CastParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  TYPE_ASSIGN_CHECK(*out_attrs, 0, param.dtype);
  return (*in_attrs)[0] != -1;
}

template<typename xpu>
void CastCompute(const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx,
                 const std::vector<TBlob>& inputs,
                 const std::vector<OpReqType>& req,
                 const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DstDType, {
    Tensor<xpu, 1, DstDType> out = outputs[0].FlatTo1D<xpu, DstDType>(s);
    MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, SrcDType, {
      Tensor<xpu, 1, SrcDType> data = inputs[0].FlatTo1D<xpu, SrcDType>(s);
      Assign(out, req[0], tcast<DstDType>(data));
    });
  });
}

struct CastStorageParam : public dmlc::Parameter<CastStorageParam> {
  // use int for enumeration
  // TODO(haibin) add enum for storage_type. Probably also aux-types
  int storage_type;
  DMLC_DECLARE_PARAMETER(CastStorageParam) {
    DMLC_DECLARE_FIELD(storage_type)
    .describe("Output storage type.");
  }
};

/*!
 * \brief This is the kernel for initializing row_idx array
 * of a RSP matrix. Each thread checks a row of the matrix,
 * if non-zero elements are found, mark this row as non-zero
 * by row_idx[cur_row_id] = cur_row_id. Otherwise,
 * row_idx[cur_row_id] = num_rows.
 */
struct FillRowIdx {
  template<typename DType, typename RType>
  MSHADOW_XINLINE static void Map(int i, RType* row_idx, const DType* arr,
                                  const int num_rows, const int num_cols) {
    row_idx[i] = num_rows;
    const int offset = i * num_cols;
    for (int j = 0; j < num_cols; ++j) {
      if (arr[offset+j] != 0) {
        row_idx[i] = i;
        break;
      }
    }
  }
};

/*!
 * \brief
 * Given a DNS storage type tensor, create a RSP type sparse tensor
 * from it. This would deep-copy non-zero rows of the dense vector to
 * the data blob of the RSP tensor and allocate memory for storing the
 * non-zero row indices in the RSP tensor.
 * TODO(junwu): The argument type for the dense ndarray is TBlob instead
 * of NDArray since it's convenient to call this function from any
 * operator's Forward/Backward functions where dev_id is unknown
 * but required to wrap a TBlob object as an NDArray. See the use case
 * in DotForwardCsrDnsRsp in matrix_op-inl.h.
 * Will revisit this interface in the future.
 */
template<typename xpu>
void CastStorageDnsRspImpl(const OpContext& ctx, const TBlob& dns, NDArray* rsp) {
  CHECK(rsp != nullptr);
  CHECK_EQ(rsp->storage_type(), kRowSparseStorage);
  CHECK_EQ(dns.shape_, rsp->shape());

  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  // TODO(junwu): allocate row_idx array for rsp with size=dns.shape[0]
  // rsp->AllocAuxData(rowsparse::kIdx, mshadow::Shape1(dns.shape_[0]));
  MSHADOW_TYPE_SWITCH(dns.type_flag_, DType, {  // data type
    NDARRAY_IDX_TYPE_SWITCH(rsp->aux_type(rowsparse::kIdx), RType, {  // row idx type
      RType* row_idx = rsp->aux_data(rowsparse::kIdx).dptr<RType>();
      const index_t num_rows = dns.shape_[0];
      const index_t num_cols = dns.shape_[1];
      // Fill input_data.shape_[0] into row_idx array
      mxnet_op::Kernel<FillRowIdx, xpu>::Launch(s, num_rows, row_idx, dns.dptr<DType>(),
          num_rows, num_cols);

      // single thread scanning row_idx array to find out number of non-zero rows
      index_t nnr = 0;  // number of non-zero rows
      for (index_t i = 0; i < num_rows; ++i) {
        if (row_idx[i] < static_cast<RType>(num_rows)) ++nnr;
      }
      if (0 == nnr) return;  // zero matrix
      // TODO(junwu): allocate data array for rsp
      // rsp->AllocData(Shape2(nnr, num_cols));
      // single thread for compressing row_idx and copying data
      // from dns to rsp, might be a bottleneck.
      auto in_tensor = dns.FlatTo2D<xpu, DType>(s);
      auto out_tensor = rsp->data().FlatTo2D<xpu, DType>(s);
      int last_nnr_id = -1;  // last non-zero row id
      for (index_t i = 0; i < num_rows; ++i) {
        if (row_idx[i] < static_cast<RType>(num_rows)) {  // non-zero row found
          row_idx[++last_nnr_id] = row_idx[i];
          mshadow::Copy(out_tensor[last_nnr_id], in_tensor[i], s);
        }
      }
      // update effective size (not capacity) of the row_idx of rsp
      rsp->SetAuxShape(rowsparse::kIdx, mshadow::Shape1(last_nnr_id+1));
    });
  });
}

template<typename xpu>
void CastStorageComputeRspDns(const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx,
                 const std::vector<NDArray>& inputs,
                 const std::vector<OpReqType>& req,
                 const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  if (inputs[0].storage_type() == kDefaultStorage
      && outputs[0].storage_type() == kRowSparseStorage) {
    NDArray tmp = outputs[0];  // get rid of the const qualifer
    CastStorageDnsRspImpl<xpu>(ctx, inputs[0].data(), &tmp);
  }
  // TODO(junwu): move the following code to a impl function
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  auto out = outputs[0];
  auto in = inputs[0];
  auto stype = in.storage_type();
  CHECK_EQ(stype, kRowSparseStorage);
  CHECK_EQ(out.storage_type(), kDefaultStorage);
  MSHADOW_TYPE_SWITCH(in.dtype(), DType, {
    MSHADOW_TYPE_SWITCH(in.aux_type(rowsparse::kIdx), IType, {
      // Fill in zeros. SLOW
      out.data().FlatTo1D<xpu, DType>(s) = 0;
      // data() is not empty
      if (in.storage_shape().ndim() != 0) {
        // Copy over
        auto in_data = in.data().FlatTo2D<xpu, DType>(s);
        auto out_data = out.data().FlatTo2D<xpu, DType>(s);
        auto num_rows = in.aux_shape(rowsparse::kIdx)[0];
        auto in_idx = in.aux_data(rowsparse::kIdx).FlatTo1D<xpu, IType>(s);
        for (size_t i = 0; i < num_rows; i += 1) {
          mshadow::Copy(out_data[in_idx[i]], in_data[i], s);
        }
      }
    });
  });
}

template<typename xpu>
void CastStorageComputeEx(const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx,
                 const std::vector<NDArray>& inputs,
                 const std::vector<OpReqType>& req,
                 const std::vector<NDArray>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(inputs.size(), 1);
  CHECK_EQ(outputs.size(), 1);
  auto stype = inputs[0].storage_type();
  auto out_stype = outputs[0].storage_type();
  if (stype == kRowSparseStorage && out_stype == kDefaultStorage) {
    CastStorageComputeRspDns<xpu>(attrs, ctx, inputs, req, outputs);
  } else {
    LOG(FATAL) << "Not implemented";
  }
}

#define MXNET_OPERATOR_REGISTER_UNARY(name)                         \
  NNVM_REGISTER_OP(name)                                            \
  .set_num_inputs(1)                                                \
  .set_num_outputs(1)                                               \
  .set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<1, 1>)  \
  .set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)     \
  .set_attr<nnvm::FInplaceOption>("FInplaceOption",                 \
    [](const NodeAttrs& attrs){                                     \
      return std::vector<std::pair<int, int> >{{0, 0}};             \
    })                                                              \
  .add_argument("data", "NDArray-or-Symbol", "The input")

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_TENSOR_ELEMWISE_UNARY_OP_H_
