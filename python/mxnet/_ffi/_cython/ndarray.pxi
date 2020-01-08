import ctypes
from ...cython.ndarray import _ndarray_cls, _np_ndarray_cls

# cdef NewArray(NDArrayHandle handle, int stype=-1, int is_np_array=0):
#     """Create a new array given handle"""
#     create_array_fn = _np_ndarray_cls if is_np_array else _ndarray_cls
#     return create_array_fn(_ctypes.cast(<unsigned long long>handle, _ctypes.c_void_p), stype=stype)


cdef c_make_array(void* handle):
    create_array_fn = _np_ndarray_cls
    return create_array_fn(ctypes.cast(<unsigned long long>handle, ctypes.c_void_p))
