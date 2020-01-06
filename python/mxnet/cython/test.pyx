include "./convert.pxi"

cdef extern from "mxnet/c_api_adt.h":
    cdef void MXTestADT(size_t ptr)

def testADT(x):
    MXTestADT(<size_t>(convert_tuple(x).get()))
    # MXTestADT(<size_t>(0))
