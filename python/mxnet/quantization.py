from __future__ import absolute_import

import numpy as np
import ctypes
from .base import _LIB, check_call
from .base import c_array, c_str, mx_uint
from .base import NDArrayHandle, SymbolHandle
from .symbol import Symbol
from . import ndarray as nd
from .ndarray import NDArray
from .io import DataIter


def quantize(param):
    max_range = nd.max(param)
    min_range = nd.min(param)
    return nd.contrib.quantize(param, min_range, max_range)


def quantize_params(qsym, params):
    inputs_name = qsym.list_arguments()
    quantized_params = {}
    for name in inputs_name:
        if name.endswith(('weight_quantize', 'bias_quantize')):
            origin_name = name.replace('_quantize', '')
            val, vmin, vmax = quantize(params[origin_name])
            quantized_params[name] = val
            quantized_params[name+'_min'] = vmin
            quantized_params[name+'_max'] = vmax
        elif name in params:
            quantized_params[name] = params[name]
    return quantized_params


def quantize_graph(sym, ignore_symbols=None, offline_params=None):
    num_ignore = 0
    ignore_handles = []
    if ignore_symbols is not None:
        assert isinstance(ignore_symbols, list)
        num_ignore = len(ignore_symbols)
        for s in ignore_symbols:
            ignore_handles.append(s.handle)

    num_offline = 0
    offline = []
    if offline_params is not None:
        num_offline = len(offline_params)
        for k in offline_params:
            offline.append(c_str(k))

    out = SymbolHandle()
    check_call(_LIB.MXQuantizeGraph(sym.handle,
                                    ctypes.byref(out),
                                    mx_uint(num_ignore),
                                    c_array(SymbolHandle, ignore_handles),
                                    mx_uint(num_offline),
                                    c_array(ctypes.c_char_p, offline)))
    return Symbol(out)


class LayerOutputQuantileCollector(object):
    def __init__(self, low_quantile=0.05, high_quantlie=0.95, include_layer=None):
        self.quantile_dict = {}
        self.low_quantile = low_quantile
        self.high_quantile = high_quantlie
        if low_quantile > high_quantlie:
            raise RuntimeError('Expected low_quantile <= high_quantile in LayerOutputQuantileCollector,'
                               'while low_quantile = %.2f and hight_quantile = %.2f'
                               % (low_quantile, high_quantlie))
        self.include_layer = include_layer

    def collect_quantiles(self, name, ndarray):
        if self.include_layer is not None and not self.include_layer(name):
            return
        handle = ctypes.cast(ndarray, NDArrayHandle)
        ndarray = NDArray(handle, writable=False)
        ndarray_np = ndarray.asnumpy().flatten()
        length = len(ndarray_np)
        if self.low_quantile == 0:
            low_th = np.nanmin(ndarray_np)
        else:
            low_idx = int(self.low_quantile * length)
            low_th = np.partition(ndarray_np, low_idx)[low_idx]
        if self.low_quantile == 1:
            high_th = np.nanmax(ndarray_np)
        else:
            high_idx = int(self.high_quantile * length)
            if high_idx == length:
                high_idx = max(length-1, 0)
            high_th = np.partition(ndarray_np, high_idx)[high_idx]
        self.quantile_dict[name] = (low_th, high_th)

    def reset(self, low_quantile=0.05, high_quantile=0.95, include_layer=None):
        self.low_quantile = low_quantile
        self.high_quantile = high_quantile
        self.include_layer = include_layer
        self.quantile_dict = {}


def calibrate_quantized_sym(qsym, quantile_dict, calib_table_type):
    if quantile_dict is None or len(quantile_dict) == 0:
        return qsym
    num_layer_outputs = len(quantile_dict)
    layer_output_names = []
    low_quantiles = []
    high_quantiles = []
    for k, v in quantile_dict.items():
        layer_output_names.append(k)
        low_quantiles.append(v[0])
        high_quantiles.append(v[1])

    calibrated_sym = SymbolHandle()
    check_call(_LIB.MXSetCalibTableToQuantizedGraph(qsym.handle,
                                                    ctypes.c_char_p(calib_table_type),
                                                    mx_uint(num_layer_outputs),
                                                    c_array(ctypes.c_char_p, layer_output_names),
                                                    c_array(ctypes.c_float, low_quantiles),
                                                    c_array(ctypes.c_float, high_quantiles),
                                                    ctypes.byref(calibrated_sym)))
    return Symbol(calibrated_sym)


def collect_layer_output_quantiles(mod, data, collector, max_num_examples=None):
    mod.set_monitor_callback(collector.collect_quantiles)
    if isinstance(data, NDArray):
        mod.forward(data_batch=data, is_train=False)
        return collector.quantile_dict
    elif isinstance(data, DataIter):
        quantile_dict = {}
        num_batches = 0
        num_examples = 0
        for batch in data:
            mod.forward(data_batch=batch, is_train=False)
            num_batches += 1
            num_examples += data.batch_size
            for k, v in collector.quantile_dict.items():
                if k in quantile_dict:
                    cur_quantiles = quantile_dict[k]
                    quantile_dict[k] = (min(cur_quantiles[0], float(v[0])), max(cur_quantiles[1], float(v[1])))
                else:
                    quantile_dict[k] = (float(v[0]), float(v[1]))
            if max_num_examples is not None and num_examples >= max_num_examples:
                break

        if num_batches == 0:
            raise RuntimeError('No batches fetched from data iter')

        return quantile_dict
