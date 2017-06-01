from __future__ import absolute_import

import ctypes
from .base import _LIB, string_types, numeric_types, check_call
from .base import c_array, py_str, c_str, mx_real_t
from .base import NDArrayHandle, ExecutorHandle, SymbolHandle
from .symbol import Symbol
from . import ndarray as nd

def quantize(param):
    max_range = nd.max(param)
    min_range = nd.min(param)
    return nd.quantize(param, min_range, max_range, out_type='uint8')

def quantize_params(sym, params):
    # params: dict of str->ndarray
    quantized_sym = quantize_graph(sym, offline_params=True)
    inputs_name = quantized_sym.list_arguments()
    quantized_params = {}
    for name in inputs_name:
        if name.endswith('_quantize'):
            origin_name = name.replace('_quantize', '')
            val, vmin, vmax = quantize(params[origin_name])
            quantized_params[name] = val
            quantized_params[name+'_min'] = vmin
            quantized_params[name+'_max'] = vmax
        elif '_quantize_' not in name:
            quantized_params[name] = params[name]
    return quantized_params

def quantize_graph(sym, offline_params=False):
    out = SymbolHandle()
    check_call(_LIB.MXQuantizeGraph(sym.handle,
                                    ctypes.byref(out),
                                    ctypes.c_int(offline_params)))
    return Symbol(out)

