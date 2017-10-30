import argparse
from common import modelzoo
import mxnet as mx
import time
import os
import logging
from mxnet.quantization import *


parser = argparse.ArgumentParser(description='score a model on a dataset')
parser.add_argument('--model', type=str, required=True,
                    help = 'the model name.')
parser.add_argument('--gpus', type=str, default='0')
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--rgb-mean', type=str, default='0,0,0')
#parser.add_argument('--data-val', type=str, required=True)
parser.add_argument('--image-shape', type=str, default='3,224,224')
parser.add_argument('--data-nthreads', type=int, default=4,
                    help='number of threads for data decoding')
parser.add_argument('--low-quantile', type=float, default=0)
parser.add_argument('--high-quantile', type=float, default=1)
args = parser.parse_args()

batch_size = args.batch_size
low_quantile = args.low_quantile
high_quantile = args.high_quantile

# number of predicted and calibrated images can be changed
num_predicted_images = batch_size * 2
num_calibrated_images = batch_size * 1

data_nthreads = args.data_nthreads
#data_val = args.data_val
gpus = args.gpus
image_shape = args.image_shape
model = args.model
rgb_mean = args.rgb_mean

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


mean_img = None
label_name = 'softmax_label'


# create data iterator
data_shape = tuple([int(i) for i in image_shape.split(',')])
if mean_img is not None:
    mean_args = {'mean_img':mean_img}
elif rgb_mean is not None:
    rgb_mean = [float(i) for i in rgb_mean.split(',')]
    mean_args = {'mean_r': rgb_mean[0], 'mean_g': rgb_mean[1], 'mean_b':rgb_mean[2]}

data_filename = 'val-5k-256.rec'
data_dirname = 'data'
data_val = data_dirname + '/' + data_filename


def download_data():
    return mx.test_utils.download(url='http://data.mxnet.io/data/val-5k-256.rec',
                                  fname=data_filename, dirname=data_dirname, overwrite=False)


print('Downloading validation dataset from http://data.mxnet.io/data/val-5k-256.rec')
download_data()

data = mx.io.ImageRecordIter(
    path_imgrec        = data_val,
    label_width        = 1,
    preprocess_threads = data_nthreads,
    batch_size         = batch_size,
    data_shape         = data_shape,
    label_name         = label_name,
    rand_crop          = False,
    rand_mirror        = False,
    **mean_args)


if isinstance(model, str):
    # download model
    print('Downloading model from MXNet model zoo')
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

print('====================================================================\n')
sym_file_name = 'resnet152-fp32-symbol.json'
print('Saving resnet152 fp32 symbol into file %s' % sym_file_name)
sym.save(sym_file_name)

print('====================================================================\n')
# cudnn int8 convolution only support channels a multiple of 4
# have to ignore quantizing conv0 node
ignore_symbols = []
ignore_sym_names = ['conv0']
for name in ignore_sym_names:
    nodes = sym.get_internals()
    idx = nodes.list_outputs().index(name + '_output')
    ignore_symbols.append(nodes[idx])

qsym = quantize_graph(sym, ignore_symbols=ignore_symbols, offline_params=arg_params.keys())
print('Saving resnet152 int8 uncalibrated symbol into file %s' % sym_file_name)
sym_file_name = 'resnet152-int8-uncalibrated-symbol.json'
qsym.save(sym_file_name)

print('====================================================================\n')
qparam_filename = 'resnet152-int8.params'
print('Saving quantized params into file %s' % qparam_filename)
qarg_params = quantize_params(qsym, arg_params)
qparam_dict = {('arg:%s' % k) : v.as_in_context(mx.cpu()) for k, v in qarg_params.items()}
# TODO: need to quantize aux params as well once BatchNorm int8 is implemented
qparam_dict.update({('aux:%s' % k) : v.as_in_context(mx.cpu()) for k, v in aux_params.items()})
mx.nd.save(qparam_filename, qparam_dict)

print('====================================================================\n')
# calibrate model by collecting quantiles from FP32 model outputs
print('Calibrating quantized model, this may take a while...')
include_layer = lambda name: name.endswith('_output')
collector = LayerOutputQuantileCollector(low_quantile=low_quantile,
                                         high_quantlie=high_quantile,
                                         include_layer=include_layer)
mod = mx.mod.Module(symbol=sym, context=devs, label_names=[label_name,])
mod.bind(for_training=False,
         data_shapes=data.provide_data,
         label_shapes=data.provide_label)
mod.set_params(arg_params, aux_params)
data.reset()
quantile_dict = mx.quantization.collect_layer_output_quantiles(mod, data, collector,
                                                               max_num_examples=num_calibrated_images)
calib_table_type = 'float32'
cqsym = mx.quantization.calibrate_quantized_sym(qsym, quantile_dict, calib_table_type)
sym_file_name = 'resnet152-int8-calibrated-symbol.json'
print('Saving resnet152 int8 calibrated symbol into file %s' % sym_file_name)
cqsym.save(sym_file_name)


