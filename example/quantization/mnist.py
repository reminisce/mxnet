import mxnet as mx
import numpy as np
import logging
from sklearn.datasets import fetch_mldata
from mxnet.quantization import *

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

data = mx.symbol.Variable('data')
fc1  = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=32, no_bias=True)
# act1 = mx.symbol.relu(data = fc1)
# fc2  = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 64)
# act2 = mx.symbol.relu(data = fc2)
# fc3  = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=10)
# mlp  = mx.symbol.SoftmaxOutput(data = fc3, name = 'softmax')
mlp  = mx.symbol.SoftmaxOutput(data = fc1, name = 'softmax')

print(mlp.list_arguments())


# prepare data
mnist = fetch_mldata('MNIST original')
np.random.seed(1234) # set seed for deterministic ordering
p = np.random.permutation(mnist.data.shape[0])
X = mnist.data[p]
Y = mnist.target[p]

X = X.astype(np.float32)/255
X_train = X[:60000]
X_test = X[60000:]
Y_train = Y[:60000]
Y_test = Y[60000:]

batch_size = 100
train_iter = mx.io.NDArrayIter(X_train, Y_train, batch_size=batch_size)
test_iter = mx.io.NDArrayIter(X_test, Y_test, batch_size=batch_size)

# training
model = mx.model.FeedForward(
    ctx = mx.gpu(0),      # Run on GPU 0
    symbol = mlp,         # Use the network we just defined
    num_epoch = 10,       # Train for 10 epochs
    learning_rate = 0.1,  # Learning rate
    momentum = 0.9,       # Momentum for SGD with momentum
    wd = 0.00001)         # Weight decay for regularization

model.fit(
    X=train_iter,  # Training data set
    eval_data=test_iter,  # Testing data set. MXNet computes scores on test set every epoch
    batch_end_callback = mx.callback.Speedometer(batch_size, 200))  # Logging module to print out progress


print 'Accuracy:', model.score(test_iter)*100, '%'


quantized_mlp = quantize_graph(mlp)
print(quantized_mlp.debug_str())


params = model.arg_params

def test(symbol):
    model = mx.model.FeedForward(
        symbol,
        ctx=mx.gpu(0),
        arg_params=params)
    print 'Accuracy:', model.score(test_iter)*100, '%'

test(mlp)
test(quantized_mlp)

ctx = mx.gpu(0)
data   = test_iter.data[0][1][:32].copyto(ctx)
weight = params['fc1_weight'].copyto(ctx)

min0d  = nd.min(data)
max0d  = nd.max(data)
qdata, min1d, max1d   = mx.contrib.nd.quantize(data, min0d, max0d)
min0w  = nd.min(weight)
max0w  = nd.max(weight)
qweight, min1w, max1w = mx.contrib.nd.quantize(weight, min0w, max0w)
qfc1, min2, max2  = nd.quantized_fully_connected(qdata, qweight,
	min1d, max1d, min1w, max1w, num_hidden=32, no_bias=True)
qfc1_, min3, max3 = nd.quantize_down_and_shrink_range(qfc1, min2, max2)

