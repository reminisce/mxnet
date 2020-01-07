#include <mxnet/runtime/ffi_helper.h>
#include <mxnet/runtime/container.h>
#include <mxnet/runtime/packed_func.h>
#include <mxnet/c_api_adt.h>
#include <mxnet/api_registry.h>
#include <iostream>

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

MXNET_REGISTER_API("test")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  std::cout << "inside test" << std::endl;
});

}  // namespace mxnet
