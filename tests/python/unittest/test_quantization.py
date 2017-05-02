import mxnet as mx
import mxnet.ndarray as nd
import numpy as np

ctx = mx.gpu(0)
dtype = np.int8
n = 4

def test_quantized_lrn():
    n = 5
    x_ = np.random.uniform(low=-100, high=100, size=(1,1,n,n))
    x = nd.array(x_, ctx=ctx, dtype=dtype)
    y = nd.quantized_lrn(x, nsize=3)

def test_quantized_convolution():
    x_ = np.random.uniform(low=-100, high=100, size=(4, 5, 5, 4))
    k_ = np.random.uniform(low=-100, high=100, size=(4, 3, 3, 4))
    x = nd.array(x_, ctx=ctx, dtype=dtype)
    k = nd.array(k_, ctx=ctx, dtype=dtype)
    min0x = nd.array([-1.0], ctx=ctx, dtype=np.float32)
    max0x = nd.array([1.0], ctx=ctx, dtype=np.float32)
    min0k = nd.array([-1.0], ctx=ctx, dtype=np.float32)
    max0k = nd.array([1.0], ctx=ctx, dtype=np.float32)
    y, min1, max1 = nd.quantized_convolution(x, k, min0x, max0x, min0k, max0k,
            stride=[1, 1], pad=[1, 1])
    y_ = y.asnumpy().astype(np.int32)

def test_quantized_relu():
    a_ = np.random.uniform(low=-100, high=100, size=(n,n))
    a = nd.array(a_, ctx=ctx, dtype=dtype)
    min0 = nd.array([-1.0], ctx=ctx, dtype=np.float32)
    max0 = nd.array([1.0], ctx=ctx, dtype=np.float32)
    b, min1, max1 = nd.quantized_relu(a, min0, max0)

def test_quantized_max_pool():
    a_ = np.random.uniform(low=-128, high=127, size=(1, 1, n, n))
    a = nd.array(a_, ctx=ctx, dtype=dtype)
    min0 = nd.array([-1.0], ctx=ctx, dtype=np.float32)
    max0 = nd.array([1.0], ctx=ctx, dtype=np.float32)
    b, min1, max1 = nd.quantized_max_pool(a, min0, max0, kernel=[2, 2])

def test_quantized_matmul():
    n = 8
    a_ = np.random.uniform(low=-100, high=100, size=(n,n))
    a = nd.array(a_, ctx=ctx, dtype=dtype)
    b_ = np.random.uniform(low=-100, high=100, size=(n,n))
    b = nd.array(b_, ctx=ctx, dtype=dtype)
    min0a = nd.array([-1.0], ctx=ctx, dtype=np.float32)
    max0a = nd.array([1.0], ctx=ctx, dtype=np.float32)
    min0b = nd.array([-1.0], ctx=ctx, dtype=np.float32)
    max0b = nd.array([1.0], ctx=ctx, dtype=np.float32)
    c, min1, max1 = nd.quantized_matmul(a, b, min0a, max0a, min0b, max0b)

if __name__ == "__main__":
    test_quantized_relu()
    test_quantized_max_pool()
    # test_quantized_lrn()
