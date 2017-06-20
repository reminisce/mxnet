/*!
 *  Copyright (c) 2017 by Contributors
 * \file cast_storage-inl.cuh
 * \brief implementation of cast_storage op on GPU
 */
#ifndef MXNET_OPERATOR_NN_CAST_STORAGE_INL_CUH_
#define MXNET_OPERATOR_NN_CAST_STORAGE_INL_CUH_

#include <mxnet/base.h>
#include <mxnet/operator.h>

namespace mxnet {
namespace op {

inline void CastStorageDnsRspImpl(mshadow::Stream<gpu>* s, const TBlob& dns, NDArray* rsp) {
#ifdef __CUDACC__
  LOG(FATAL) << "CastStorageDnsRspImpl gpu version is not implemented.";
#endif  // __CUDACC__
}

inline void CastStorageDnsCsrImpl(mshadow::Stream<gpu>* s, const TBlob& dns, NDArray* csr) {
#ifdef __CUDACC__
  LOG(FATAL) << "CastStorageDnsCsrImpl gpu version is not implemented.";
#endif  // __CUDACC__
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NN_CAST_STORAGE_INL_CUH_
