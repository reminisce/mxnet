import ctypes
import mxnet as mx
from mxnet.base import SymbolHandle, check_call, _LIB, mx_uint, c_str_array
from mxnet.symbol import Symbol
import numpy as np


def test_subgraph():
    def get_graph():
        data1 = mx.sym.Variable('data1', shape=(3, 3, 10, 10), dtype=np.float32)
        data2 = mx.sym.Variable('data2', shape=(1, 0, 2, 2))
        data3 = mx.sym.sin(data2)
        conv = mx.sym.Convolution(data=data1, weight=data3, kernel=(2, 2), num_filter=1)
        rets = []
        rets.append((conv, []))
        rets.append((conv, [mx.sym.sin.__name__]))
        rets.append((conv, [mx.sym.Convolution.__name__]))
        rets.append((conv, [mx.sym.sin.__name__, mx.sym.Convolution.__name__]))
        return rets

    for regular_sym, op_names in get_graph():
        input_names = regular_sym.list_inputs()
        shapes = regular_sym.infer_shape()
        types = regular_sym.infer_type()
        out = SymbolHandle()

        check_call(_LIB.MXPartitionGraph(regular_sym.handle, mx_uint(len(op_names)),
            c_str_array(op_names), ctypes.byref(out)))
        subgraph_sym = Symbol(out)
        assert input_names == subgraph_sym.list_inputs()

        print(subgraph_sym.list_outputs())
        assert shapes == subgraph_sym.infer_shape()
        assert types == subgraph_sym.infer_type()

        regular_exec = regular_sym.simple_bind(ctx=mx.cpu(), grad_req='null')
        subgraph_exec = subgraph_sym.simple_bind(ctx=mx.cpu(), grad_req='null')

        for name in input_names:
            regular_exec.arg_dict[name][:] = mx.nd.random.normal(
                    shape=regular_exec.arg_dict[name].shape)
            subgraph_exec.arg_dict[name][:] = regular_exec.arg_dict[name]

        subgraph_exec.forward()
        regular_exec.forward()
        mx.nd.waitall()
        assert (subgraph_exec.outputs[0] - regular_exec.outputs[0]).abs().sum().asscalar() == 0.0


def test_input_name_order():
    def check_input_order(sym, op_names):
        out = SymbolHandle()
        check_call(_LIB.MXPartitionGraph(sym.handle, mx_uint(len(op_names)),
                                         c_str_array(op_names), ctypes.byref(out)))

        new_sym = Symbol(out)
        print(sym.list_inputs())
        print(new_sym.list_inputs())
        assert new_sym.list_inputs() == sym.list_inputs()

    def test_network_structure_1():
        data1 = mx.sym.var('data1')
        data2 = mx.sym.var('data2')
        conv1 = mx.sym.Convolution(data=data1, weight=data2, no_bias=True, kernel=(2, 2), num_filter=1)
        conv2 = mx.sym.Convolution(data=data2, weight=data1, no_bias=True, kernel=(2, 2), num_filter=1)
        out = mx.sym.Group([conv1, conv2])
        check_input_order(out, ['Convolution'])

    test_network_structure_1()


if __name__ == '__main__':
    import nose
    nose.runmodule()
