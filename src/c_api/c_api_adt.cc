#include <mxnet/runtime/ffi_helper.h>
#include <mxnet/runtime/container.h>
#include <mxnet/runtime/packed_func.h>
#include <mxnet/c_api_adt.h>
#include <mxnet/api_registry.h>
#include <mxnet/base.h>
#include <nnvm/c_api.h>
#include <iostream>

#include "../operator/tensor/init_op.h"
#include "../imperative/imperative_utils.h"

void MXTestADT(size_t ptr) {
  using namespace mxnet::runtime;
  // const Object*
  // ADT::get()
  // ADTBuilder builder = ADTBuilder(0, 10);
  // for (uint32_t i = 0; i < 10; ++i) {
  //   builder.EmplaceInit(i, Integer(i));
  // }
  // ADT adt = builder.Get();
  // const ::mxnet::runtime::ADTObj* obj = adt.as<ADTObj>();

  // const ADTObj* obj = reinterpret_cast<const ADTObj*>(ptr);
  // std::cout << "size = " << obj->size;
  // for (uint32_t i = 0; i < obj->size; ++i) {
  //   int64_t value = obj->operator[](i).as<::mxnet::runtime::IntegerObj>()->value;
  //   std::cout << value;
  // }
  
  MXNetRetValue rv;
  const PackedFunc* fp = Registry::Get("test");
  if (fp != nullptr) {
    fp->CallPacked(MXNetArgs(nullptr, nullptr, 0), &rv);
  } else {
    std::cout << "null PackedFunc" << std::endl;
  }
}

namespace mxnet {

MXNET_REGISTER_API("np.zeros1")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  const nnvm::Op* op = Op::Get("_npi_zeros");
  const runtime::ObjectRef ref = args[0].operator runtime::ObjectRef();
  const runtime::ADTObj* obj = ref.as<runtime::ADTObj>();
  // std::cout << "size = " << obj->size << std::endl;
  // for (uint32_t i = 0; i < obj->size; ++i) {
  //   int64_t value = obj->operator[](i).as<::mxnet::runtime::IntegerObj>()->value;
  //   std::cout << value << " ";
  // }
  // std::cout << std::endl;
  mxnet::op::InitOpParam param;
  param.shape = TShape(obj->size, 0);
  for (uint32_t i = 0; i < obj->size; ++i) {
    int64_t value = obj->operator[](i).as<::mxnet::runtime::IntegerObj>()->value;
    param.shape[i] = value;
  }
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
  ndoutputs[0] = reinterpret_cast<mxnet::NDArray*>(new mxnet::NDArray());
  auto state = mxnet::Imperative::Get()->Invoke(Context::CPU(), attrs, ndinputs, ndoutputs);
  
  *ret = reinterpret_cast<mxnet::NDArray*>(ndoutputs[0]);
});

MXNET_REGISTER_API("np.zeros0")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  *ret = static_cast<int64_t>(0xdeadbeaf);
});

}  // namespace mxnet
