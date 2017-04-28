import mxnet as mx
import mxnet.ndarray as nd
import numpy as np

ctx = mx.gpu(0)
dtype = np.uint8
n = 4

def test_quantized_relu():
    a_ = np.random.uniform(low=-100, high=100, size=(n,n))
    a = nd.array(a_, ctx=ctx, dtype=dtype)
    b = nd.quantized_relu(a)

def test_quantized_max_pool():
    a_ = np.random.uniform(low=-128, high=127, size=(1, 1, n, n))
    a = nd.array(a_, ctx=ctx, dtype=dtype)
    b = nd.quantized_max_pool(a, kernel=[2, 2])

if __name__ == "__main__":
    test_quantized_relu()
    test_quantized_max_pool()
