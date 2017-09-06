import argparse
from common import modelzoo, find_mxnet
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
parser.add_argument('--data-val', type=str, required=True)
parser.add_argument('--image-shape', type=str, default='3,224,224')
parser.add_argument('--data-nthreads', type=int, default=4,
                    help='number of threads for data decoding')
args = parser.parse_args()

batch_size = args.batch_size
###########################################
batch_size = 32
num_predicted_images = batch_size * 2
num_calibrated_images = batch_size * 1
low_quantile = 0
high_quantile = 1
###########################################

data_nthreads = args.data_nthreads
data_val = args.data_val
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
    mean_args = {'mean_r':rgb_mean[0], 'mean_g':rgb_mean[1],
      'mean_b':rgb_mean[2]}

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


def score(sym, arg_params, aux_params,
          data, devs, label_name, max_num_examples):
    metrics = [mx.metric.create('acc'),
               mx.metric.create('top_k_accuracy', top_k = 5)]
    if not isinstance(metrics, list):
        metrics = [metrics,]
    mod = mx.mod.Module(symbol=sym, context=devs, label_names=[label_name,])
    mod.bind(for_training=False,
             data_shapes=data.provide_data,
             label_shapes=data.provide_label)
    mod.set_params(arg_params, aux_params)

    print('Starting inference')
    tic = time.time()
    num = 0
    for batch in data:
        mod.forward(batch, is_train=False)
        for m in metrics:
            mod.update_metric(m, batch.label)
        num += batch_size
        if max_num_examples is not None and num >= max_num_examples:
            break

    speed = num / (time.time() - tic)
    print('Finished inference')

    logging.info('Finished with %f images per second', speed)
    for m in metrics:
        logging.info(m.get())


def advance_data_iter(data_iter, n):
    assert n >= 0
    if n == 0:
        return data_iter
    has_next_batch = True
    while has_next_batch:
        try:
            data_iter.next()
            n -= 1
            if n == 0:
                return data_iter
        except StopIteration:
            has_next_batch = False


print('\n\n')
print('origin model:')
data.reset()
# make sure that fp32 inference works on the same images as calibrated quantized model
data = advance_data_iter(data, num_calibrated_images/batch_size)
score(sym, arg_params, aux_params, data, devs, label_name, max_num_examples=num_predicted_images)
data.reset()
# print('symbol:')
# print(sym.debug_str())
# print('\n\n')

ignore_symbols = []
ignore_sym_names = ['conv0']
for name in ignore_sym_names:
    nodes = sym.get_internals()
    idx = nodes.list_outputs().index(name + '_output')
    ignore_symbols.append(nodes[idx])
qsym = quantize_graph(sym,
                      ignore_symbols=ignore_symbols,
                      offline_params=arg_params.keys())
qarg_params = quantize_params(qsym, arg_params)

print('after quantization:')
# print(qsym.debug_str())
# print('\n\n')
# print(arg_params)
# print(qarg_params)
# print('\n\n')
data.reset()
# make sure that int8 uncalibrated inference works on the same images as calibrated quantized model
data = advance_data_iter(data, num_calibrated_images/batch_size)
score(qsym, qarg_params, aux_params, data, devs, label_name, max_num_examples=num_predicted_images)
data.reset()
print('====================================================================\n')

#########################################################################################################
# calibrate model
print('begin collecting quantiles from quantized sym')
include_layer = lambda name: name.startswith('quantized_') and name.endswith('_out')
collector = LayerOutputQuantileCollector(low_quantile=low_quantile,
                                         high_quantlie=high_quantile,
                                         include_layer=include_layer)
mod = mx.mod.Module(symbol=qsym, context=devs, label_names=[label_name,])
mod.bind(for_training=False,
         data_shapes=data.provide_data,
         label_shapes=data.provide_label)
mod.set_params(qarg_params, aux_params)
data.reset()
quantile_dict = mx.quantization.collect_layer_output_quantiles(mod, data, collector, max_num_examples=num_calibrated_images)
data.reset()
data = advance_data_iter(data, num_calibrated_images/batch_size)
calib_table_type = 'int32'
print('finished collecting %d layer output quantiles' % len(quantile_dict))
print('begin calibrating quantized sym')
cqsym = mx.quantization.calibrate_quantized_sym(qsym, quantile_dict, calib_table_type)
print('finished calibrating quantized sym')
print('after calibration')
#print(cqsym.debug_str())
print('begin scoring calibrated quantized sym with int32 calib table')
score(cqsym, qarg_params, aux_params, data, devs, label_name, max_num_examples=num_predicted_images)
print('done scoring calibrated quantized sym with int32 calib table')
data.reset()
print('====================================================================\n')
#########################################################################################################

print('begin collecting quantiles from original sym')
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
quantile_dict = mx.quantization.collect_layer_output_quantiles(mod, data, collector, max_num_examples=num_calibrated_images)
data.reset()
data = advance_data_iter(data, num_calibrated_images/batch_size)
new_quantile_dict = {}
# keys is something like 'stage4_unit3_conv3_output'
# need to trim the last three letters so that it could
# be matched with quantized op's output name
for k, v in quantile_dict.items():
    new_quantile_dict[k[:-3]] = v
quantile_dict = new_quantile_dict
calib_table_type = 'float32'
print('finished collecting %d layer output quantiles' % len(quantile_dict))
print('begin calibrating quantized sym')
cqsym = mx.quantization.calibrate_quantized_sym(qsym, quantile_dict, calib_table_type)
print('finished calibrating quantized sym')
print('after calibration')
#print(cqsym.debug_str())
print('begin scoring calibrated quantized sym with float32 calib table')
score(cqsym, qarg_params, aux_params, data, devs, label_name, max_num_examples=num_predicted_images)
data.reset()
print('done scoring calibrated quantized sym with float32 calib table')

