import mxnet as mx
from mxnet import np

print("tvm ffi...")
a = np.zeros1((3, 4))
print(a)
print("legacy ffi...")
a = np.zeros((3, 4))
print(a)
