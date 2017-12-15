import argparse
import ctypes
import os
import logging
from mxnet.base import NDArrayHandle
from mxnet.ndarray import NDArray
from common import modelzoo
import mxnet as mx


class LayerOutputCollector(object):
    def __init__(self, include_layer=None):
        self.nd_dict = {}
        self.include_layer = include_layer

    def collect_output(self, name, ndarray):
        if self.include_layer is not None and not self.include_layer(name):
            return
        handle = ctypes.cast(ndarray, NDArrayHandle)
        ndarray = NDArray(handle, writable=False).copyto(mx.cpu())
        if name in self.nd_dict:
            self.nd_dict[name].append(ndarray)
        else:
            self.nd_dict[name] = [ndarray]

    def reset(self):
        self.nd_dict = {}
        self.include_layer = None


parser = argparse.ArgumentParser(description='score a model on a dataset')
parser.add_argument('--model', type=str, required=True,
                    help='the model name.')
parser.add_argument('--gpus', type=str, default='0')
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--rgb-mean', type=str, default='0,0,0')
parser.add_argument('--data-val', type=str, required=True)
parser.add_argument('--image-shape', type=str, default='3,224,224')
parser.add_argument('--data-nthreads', type=int, default=4,
                    help='number of threads for data decoding')
args = parser.parse_args()

batch_size = args.batch_size
data_nthreads = args.data_nthreads
data_val = args.data_val
gpus = args.gpus
image_shape = args.image_shape
model = args.model
rgb_mean = args.rgb_mean

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# number of predicted and calibrated images can be changed
max_num_calib_batches = 10
num_calib_batches = 5
assert num_calib_batches <= max_num_calib_batches
num_infer_batches = 500
num_infer_image_offset = batch_size * max_num_calib_batches
num_predicted_images = batch_size * num_infer_batches
num_calibrated_images = batch_size * num_calib_batches

mean_img = None
label_name = 'softmax_label'

# create data iterator
data_shape = tuple([int(i) for i in image_shape.split(',')])
if mean_img is not None:
    mean_args = {'mean_img': mean_img}
elif rgb_mean is not None:
    rgb_mean = [float(i) for i in rgb_mean.split(',')]
    mean_args = {'mean_r': rgb_mean[0], 'mean_g': rgb_mean[1],
                 'mean_b': rgb_mean[2]}

data = mx.io.ImageRecordIter(path_imgrec=data_val,
                             label_width=1,
                             preprocess_threads=data_nthreads,
                             batch_size=batch_size,
                             data_shape=data_shape,
                             label_name=label_name,
                             rand_crop=False,
                             rand_mirror=False,
                             **mean_args)

if isinstance(model, str):
    # download model
    dir_path = os.path.dirname(os.path.realpath(__file__))
    (prefix, epoch) = modelzoo.download_model(
        model, os.path.join(dir_path, 'model'))
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
elif isinstance(model, tuple) or isinstance(model, list):
    assert len(model) == 3
    (sym, arg_params, aux_params) = model
else:
    raise TypeError('model type [%s] is not supported' % str(type(model)))

# create module
if gpus == '':
    devs = mx.cpu()
else:
    devs = [mx.gpu(int(i)) for i in gpus.split(',')]

#include_layer = lambda name: name.find('relu') >= 0
#include_layer = lambda name: name.startswith('stage1_unit1_conv')
include_layer = lambda name: name.startswith('fc1')
collector = LayerOutputCollector(include_layer)
mod = mx.mod.Module(symbol=sym, context=devs, label_names=[label_name, ])
mod.bind(for_training=False,
         data_shapes=data.provide_data,
         label_shapes=data.provide_label)
mod.set_params(arg_params, aux_params)
data.reset()
num_batches = 0
num_examples = 0
mod.set_monitor_callback(collector.collect_output)
for batch in data:
    print('collecting ndarray from batch %d' % num_batches)
    mod.forward(data_batch=batch, is_train=False)
    num_batches += 1
    num_examples += data.batch_size
    if num_batches >= max_num_calib_batches:
        break

for k, v in collector.nd_dict.items():
    fname = k + '.nds'
    print('saving layer output ndarray to %s' % fname)
    mx.nd.save(fname, v)
