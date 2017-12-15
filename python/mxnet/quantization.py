from __future__ import absolute_import

from scipy import stats
import numpy as np
import ctypes
from .base import _LIB, check_call
from .base import c_array, c_str, mx_uint
from .base import NDArrayHandle, SymbolHandle
from .symbol import Symbol
from . import ndarray as nd
from .ndarray import NDArray
from .io import DataIter
from .context import cpu


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


class LayerOutputCollector(object):
    """Saves layer output NDArray in a dict with layer name as keys and lists of NDArrays as values."""
    def __init__(self, include_layer=None, logger=None):
        self.nd_dict = {}
        self.include_layer = include_layer
        self.logger = logger

    def collect(self, name, ndarray):
        if self.include_layer is not None and not self.include_layer(name):
            return
        handle = ctypes.cast(ndarray, NDArrayHandle)
        ndarray = NDArray(handle, writable=False).copyto(cpu())
        if self.logger is not None:
            self.logger.info("Collecting layer %s output of shape %s" % (name, ndarray.shape))
        if name in self.nd_dict:
            self.nd_dict[name].append(ndarray)
        else:
            self.nd_dict[name] = [ndarray]

    def reset(self):
        self.nd_dict = {}
        self.include_layer = None


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

    def collect(self, name, ndarray):
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


def calibrate_quantized_sym(qsym, th_dict, calib_table_type='float32'):
    if th_dict is None or len(th_dict) == 0:
        return qsym
    num_layer_outputs = len(th_dict)
    layer_output_names = []
    low_quantiles = []
    high_quantiles = []
    for k, v in th_dict.items():
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
    mod.set_monitor_callback(collector.collect)
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
    else:
        raise TypeError('collect_layer_output_quantiles only supports input of'
                        ' type NDArray and DataIter, received type=%s' % str(type(data)))


def collect_layer_outputs(mod, data, include_layer=None, max_num_examples=None, logger=None):
    collector = LayerOutputCollector(include_layer=include_layer, logger=logger)
    mod.set_monitor_callback(collector.collect)
    if isinstance(data, NDArray):
        mod.forward(data_batch=data, is_train=False)
        return collector.nd_dict
    elif isinstance(data, DataIter):
        num_examples = 0
        for batch in data:
            mod.forward(data_batch=batch, is_train=False)
            num_examples += data.batch_size
            if max_num_examples is not None and num_examples >= max_num_examples:
                break
        if num_examples == 0:
            raise RuntimeError('No examples fetched from data iter')
        return collector.nd_dict
    else:
        raise TypeError('collect_layer_output only supports input of type NDArray and DataIter, received type=%s'
                        % str(type(data)))


def _smooth_distribution(p, eps=0.0001):
    """Given a discrete distribution (might not have been normalized to 1),
    smooth it by replacing zeros with eps and taking the corresponding amount
    off the non-zero values."""
    is_zeros = (p == 0).astype(np.float32)
    is_nonzeros = (p != 0).astype(np.float32)
    n_zeros = is_zeros.sum()
    n_nonzeros = p.size - n_zeros
    eps1 = eps * float(n_zeros) / float(n_nonzeros)
    assert eps1 < 1.0, 'n_zeros=%d, n_nonzeros=%d, eps1=%f' % (n_zeros, n_nonzeros, eps1)
    hist = p.astype(np.float32)
    hist += eps * is_zeros + (-eps1) * is_nonzeros
    assert (hist <= 0).sum() == 0
    return hist


def _get_optimal_threshold(arr, num_bins=8001, num_quantized_bins=255):
    """Given a dataset, find the optimal threshold for quantizing it.
    Ref: http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf"""
    if isinstance(arr, NDArray):
        arr = arr.asnumpy()
    elif isinstance(arr, list):
        assert len(arr) != 0
        for i, nd in enumerate(arr):
            if isinstance(nd, NDArray):
                arr[i] = nd.asnumpy()
            elif not isinstance(nd, np.ndarray):
                raise TypeError('get_optimal_threshold only supports input type of NDArray,'
                                ' list of np.ndarrays or NDArrays, and np.ndarray,'
                                ' while received type=%s' % (str(type(nd))))
        arr = np.concatenate(arr)
    elif not isinstance(arr, np.ndarray):
        raise TypeError('get_optimal_threshold only supports input type of NDArray, list of NDArrays and np.ndarray,'
                        ' while received type=%s' % (str(type(arr))))
    min_val = np.min(arr)
    max_val = np.max(arr)
    th = max(abs(min_val), abs(max_val))

    hist, hist_edeges = np.histogram(arr, bins=num_bins, range=(-th, th))
    zero_bin_idx = num_bins / 2
    num_half_quantized_bins = num_quantized_bins / 2
    assert np.allclose(hist_edeges[zero_bin_idx] + hist_edeges[zero_bin_idx + 1], 0, rtol=1e-5, atol=1e-7)

    thresholds = np.zeros(num_bins / 2 + 1 - num_quantized_bins / 2)
    divergence = np.zeros_like(thresholds)
    quantized_bins = np.zeros(num_quantized_bins, dtype=np.int32)
    for i in range(num_quantized_bins / 2,
                   num_bins / 2 + 1):  # i means the number of bins on half axis excluding the zero bin
        p_bin_idx_start = zero_bin_idx - i
        p_bin_idx_stop = zero_bin_idx + i + 1
        thresholds[i - num_half_quantized_bins] = hist_edeges[p_bin_idx_stop]
        # sliced_nd_hist is used to generate candidate distribution q
        sliced_nd_hist = hist[p_bin_idx_start:p_bin_idx_stop]

        # generate reference distribution p
        p = sliced_nd_hist.copy()
        assert p.size % 2 == 1
        assert p.size >= num_quantized_bins
        # put left outlier count in p[0]
        left_outlier_count = np.sum(hist[0:p_bin_idx_start])
        p[0] += left_outlier_count
        # put right outlier count in p[-1]
        right_outlier_count = np.sum(hist[p_bin_idx_stop:])
        p[-1] += right_outlier_count
        # is_nonzeros[k] indicates whether hist[k] is nonzero
        is_nonzeros = (sliced_nd_hist != 0).astype(np.int32)

        # calculate how many bins should be merged to generate quantized distribution q
        num_merged_bins = p.size / num_quantized_bins
        # merge hist into num_quantized_bins bins
        for j in range(num_quantized_bins):
            start = j * num_merged_bins
            stop = start + num_merged_bins
            quantized_bins[j] = sliced_nd_hist[start:stop].sum()
        quantized_bins[-1] += sliced_nd_hist[num_quantized_bins * num_merged_bins:].sum()
        # expand quantized_bins into p.size bins
        q = np.zeros(p.size, dtype=np.float32)
        for j in range(num_quantized_bins):
            start = j * num_merged_bins
            if j == num_quantized_bins - 1:
                stop = -1
            else:
                stop = start + num_merged_bins
            norm = is_nonzeros[start:stop].sum()
            if norm != 0:
                q[start:stop] = float(quantized_bins[j]) / float(norm)
        q[sliced_nd_hist == 0] = 0
        p = _smooth_distribution(p)
        q = _smooth_distribution(q)
        divergence[i - num_half_quantized_bins] = stats.entropy(p, q)
        quantized_bins[:] = 0

    min_divergence_idx = np.argmin(divergence)
    min_divergence = divergence[min_divergence_idx]
    opt_th = thresholds[min_divergence_idx]
    return min_val, max_val, min_divergence, opt_th


def get_optimal_thresholds(nd_dict, num_bins=8001, num_quantized_bins=255, logger=None):
    """Given a ndarray dict, find the optimal threshold for quantizing each value of the key."""
    assert isinstance(nd_dict, dict)
    if logger is not None:
        logger.info('Calculating optimal thresholds for quantization using KL divergence'
                    ' with num_bins=%d and num_quantized_bins=%d' % (num_bins, num_quantized_bins))
    th_dict = {}
    for k, v in nd_dict.items():
        min_val, max_val, min_divergence, opt_th = _get_optimal_threshold(v, num_bins=num_bins,
                                                                          num_quantized_bins=num_quantized_bins)
        th_dict[k] = (-opt_th, opt_th)
        if logger is not None:
            logger.info('layer=%s, min_val=%f, max_val=%f, min_divergence=%f, optimal_threshold=%f'
                        % (k, min_val, max_val, min_divergence, opt_th))
    return th_dict
