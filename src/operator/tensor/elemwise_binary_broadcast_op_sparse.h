/*!
 *  Copyright (c) 2017 by Contributors
 * \file elementwise_binary_broadcast_sparse_op.h
 * \brief Function defintion of elementwise sparse operators
 */
#ifndef MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_BROADCAST_OP_SPARSE_H_
#define MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_BROADCAST_OP_SPARSE_H_

#include <mxnet/operator_util.h>
#include <algorithm>
#include <vector>
#include <string>
#include <utility>
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"
#include "./elemwise_binary_op.h"
#include "../operator_common.h"
#include "../mxnet_op.h"

namespace mxnet {
namespace op {

/*!
 * \brief Kernel of sparse (m, n) element_wise dense (m, 1) or (1, n)
 * \tparam is_col_vector indicates whether the dense vector is col or row
 */
template<typename OP, bool is_col_vector>
struct BroadcastOpCsrDenseVector {
  /*!
   * TODO(junwu): A better looping mechanism is loop
   * through every nnz. For each nnz, find its corresponding
   * row id (maybe a binary search). This one loop through
   * rows instead. May need to change in the future.
   * \brief The Map handles one row of the matrix
   * \param i indicates the i-th row of the matrix
   */
  template<typename DType, typename IType, typename CType>
  MSHADOW_XINLINE static void Map(int i, DType* data, IType* indptr, CType* col_idx,
                                  const DType* data_l, const IType* indptr_l,
                                  const CType* col_idx_l, const DType* data_r) {
    indptr[i+1] = indptr_l[i+1];
    for (IType j = indptr_l[i]; j < indptr_l[i+1]; ++j) {
      col_idx[j] = col_idx_l[j];
      const int k = (is_col_vector? i : col_idx_l[j]);
      data[j] = OP::Map(data_l[j],  data_r[k]);
    }
  }
};

// TODO(junwu): only support csr lhs, dense rhs, csr sparse ret for now
// make it more generic later
// current sparse shape is (m, n), dense shape is (m, 1), or (1, n).
// The op must satisfy the following condition:
// op(0, value) = 0. An example is op=mul.
template<typename xpu, typename OP>
void BinaryBroadcastComputeCsrDense(const nnvm::NodeAttrs& attrs,
                                    const OpContext& ctx,
                                    const std::vector<NDArray>& inputs,
                                    const std::vector<OpReqType>& req,
                                    const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs.size(), 2);
  CHECK_EQ(outputs.size(), 1);

  const NDArray& nd_l = inputs[0];
  const NDArray& nd_r = inputs[1];
  const NDArray& ret = outputs[0];

  // only support csr nd_l, dense nd_r, and sparse ret for now
  CHECK_EQ(nd_l.storage_type(), kCSRStorage)
    << "BinaryBroadcastComputeCsrDe only supports csr left";
  CHECK_EQ(nd_r.storage_type(), kDefaultStorage)
    << "BinaryBroadcastComputeCsrDe only supports dense right";
  CHECK_EQ(ret.storage_type(), kCSRStorage)
    << "BinaryBroadcastComputeCsrDe only supports csr ret";

  // TODO(junwu): check left and right data types are consistent

  const auto num_rows = nd_l.shape()[0];
  const auto num_cols = nd_l.shape()[1];
  const auto num_rows_r = nd_r.shape()[0];
  const auto num_cols_r = nd_r.shape()[1];

  // only support broadcast dense on the right hand side
  CHECK(num_rows == num_rows_r || num_rows_r == 1);
  CHECK(num_cols == num_cols_r || num_cols_r == 1);

  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(ret.dtype(), DType, {
    NDARRAY_IDX_TYPE_SWITCH(nd_l.aux_type(csr::kIndPtr), IType, {
      NDARRAY_IDX_TYPE_SWITCH(nd_l.aux_type(csr::kIdx), CType, {
        const DType* data_l = nd_l.data().dptr<DType>();
        const IType* indptr_l = nd_l.aux_data(csr::kIndPtr).dptr<IType>();
        const CType* col_idx_l = nd_l.aux_data(csr::kIdx).dptr<CType>();
        const DType* data_r = nd_r.data().dptr<DType>();

        // allocate ret based on the storage shape of nd_l
        // may need further compression after performing the op
        ret.CheckAndAlloc({TShape({num_rows+1}), TShape({nd_l.data().Size()})});
        DType* data = ret.data().dptr<DType>();
        IType* indptr = nd_l.aux_data(csr::kIndPtr).dptr<IType>();
        CType* col_idx = nd_l.aux_data(csr::kIdx).dptr<CType>();
        indptr[0] = 0;

        if (num_cols_r == 1) {  // (m, n) x (m, 1)
          mxnet_op::Kernel<BroadcastOpCsrDenseVector<OP, true>, xpu>::Launch(s, num_rows,
              data, indptr, col_idx, data_l, indptr_l, col_idx_l, data_r);
        } else if (num_rows_r == 1) {  // (m, n) x (1, n)
          mxnet_op::Kernel<BroadcastOpCsrDenseVector<OP, false>, xpu>::Launch(s, num_rows,
              data, indptr, col_idx, data_l, indptr_l, col_idx_l, data_r);
        } else {
          LOG(FATAL) << "Not implemented for BroadcastOp(Csr(m, n), de(m, n))";
        }
      });
    });
  });

  // TODO(junwu): may need to compress the sparse out ndarray as there may be
  // zero elements after broadcast_xxx
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_BROADCAST_OP_SPARSE_H_
