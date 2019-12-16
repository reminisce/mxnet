//#if MXNET_USE_PYBIND11
#include <inttypes.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <mxnet/base.h>
#include "../../operator/tensor/init_op.h"
#include "../../imperative/imperative_utils.h"

namespace py = ::pybind11;

size_t _npi_zeros(size_t op_handle, const std::vector<int>& shape) {
  const nnvm::Op* op = static_cast<nnvm::Op*>(reinterpret_cast<void*>(op_handle));

  mxnet::op::InitOpParam param;
  param.shape = TShape(shape.begin(), shape.end());
  param.dtype = 0;
  param.ctx = "cpu";
  nnvm::NodeAttrs attrs;
  attrs.parsed = std::move(param);
  attrs.op = op;

  int num_inputs = 0;
  int infered_num_outputs;
  int num_visible_outputs;
  mxnet::imperative::SetNumOutputs(op, attrs, num_inputs, &infered_num_outputs, &num_visible_outputs);

  std::vector<mxnet::NDArray*> ndoutputs(1, nullptr), ndinputs;
  ndoutputs[0] = new mxnet::NDArray();
  auto state = mxnet::Imperative::Get()->Invoke(Context::CPU(), attrs, ndinputs, ndoutputs);

  return reinterpret_cast<size_t>(ndoutputs[0]);
}

size_t _npi_zeros_dummy(size_t op_handle, const std::vector<int>& shape) {
  return 0;
}

PYBIND11_MODULE(example, m) {
  m.def("_npi_zeros", &_npi_zeros, "Creating zeros in shape");
  m.def("_npi_zeros_dummy", &_npi_zeros_dummy, "Creating zeros in shape");
}

//#endif  // MXNET_USE_PYBIND11
