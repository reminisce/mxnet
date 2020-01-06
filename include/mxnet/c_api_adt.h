#ifndef MXNET_C_API_ADT_H_
#define MXNET_C_API_ADT_H_

/*! \brief Inhibit C++ name-mangling for MXNet functions. */
#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#include "c_api.h"

MXNET_DLL void MXTestADT(size_t ptr);

#ifdef __cplusplus
}
#endif  // __cplusplus
#endif  // MXNET_C_API_ADT_H_
