"""resnet calibration using kl divergence"""
import argparse
from common import modelzoo
import mxnet as mx
import time
import os
import logging
from mxnet.quantization import *

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
parser.add_argument('--num-calib-batches', type=int, default=10,
                    help='number of batches for calibration')
parser.add_argument('--max-num-calib-batches', type=int, default=50,
                    help='max allowed number of batches for calibration')
parser.add_argument('--ignore-last-fc', action='store_true',
                    help='use fp32 for the last fc node')
parser.set_defaults(ignore_last_fc=False)
args = parser.parse_args()

batch_size = args.batch_size

# number of predicted and calibrated images can be changed
num_calib_batches = args.num_calib_batches
max_num_calib_batches = args.max_num_calib_batches
assert num_calib_batches <= max_num_calib_batches

num_infer_batches = 500
num_infer_image_offset = batch_size * max_num_calib_batches
num_predicted_images = batch_size * num_infer_batches
num_calibrated_images = batch_size * num_calib_batches

data_nthreads = args.data_nthreads
data_val = args.data_val
gpus = args.gpus
image_shape = args.image_shape
model = args.model
rgb_mean = args.rgb_mean
ignore_last_fc = args.ignore_last_fc

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

mean_img = None
label_name = 'softmax_label'

# create data iterator
data_shape = tuple([int(i) for i in image_shape.split(',')])
if mean_img is not None:
    mean_args = {'mean_img': mean_img}
elif rgb_mean is not None:
    rgb_mean = [float(i) for i in rgb_mean.split(',')]
    mean_args = {'mean_r': rgb_mean[0], 'mean_g': rgb_mean[1], 'mean_b': rgb_mean[2]}

# data_filename = 'val-5k-256.rec'
data_filename = 'val_256_q90.rec'
data_dirname = 'data'
data_val = data_dirname + '/' + data_filename
url = 'http://data.mxnet.io/data/' + data_filename


def download_data():
    return mx.test_utils.download(url=url, fname=data_filename, dirname=data_dirname, overwrite=False)


logger.info('Downloading validation dataset from %s' % url)
download_data()

data = mx.io.ImageRecordIter(
    path_imgrec=data_val,
    label_width=1,
    preprocess_threads=data_nthreads,
    batch_size=batch_size,
    data_shape=data_shape,
    label_name=label_name,
    rand_crop=False,
    rand_mirror=False,
    shuffle=True,
    shuffle_chunk_seed=3982304,
    seed=48564309,
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


def score(sym, arg_params, aux_params, data, devs, label_name, max_num_examples):
    metrics = [mx.metric.create('acc'),
               mx.metric.create('top_k_accuracy', top_k=5)]
    if not isinstance(metrics, list):
        metrics = [metrics, ]
    mod = mx.mod.Module(symbol=sym, context=devs, label_names=[label_name, ])
    mod.bind(for_training=False,
             data_shapes=data.provide_data,
             label_shapes=data.provide_label)
    mod.set_params(arg_params, aux_params)

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

    logging.info('Finished inference with %d images' % num)
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


# cudnn int8 convolution only support channels a multiple of 4
# have to ignore quantizing conv0 node
ignore_symbols = []
ignore_sym_names = ['conv0']
if ignore_last_fc:
    ignore_sym_names.append('fc1')
for name in ignore_sym_names:
    nodes = sym.get_internals()
    idx = nodes.list_outputs().index(name + '_output')
    ignore_symbols.append(nodes[idx])

logger.info('Quantizing the FP32 model...')
qsym = quantize_graph(sym, ignore_symbols=ignore_symbols, offline_params=arg_params.keys())
logger.info('Finished quantizing the FP32 model')

logger.info('Quantizing parameters of the FP32 model...')
qarg_params = quantize_params(qsym, arg_params)
logger.info('Finished quantizing the parameters of the FP32 model')

logger.info('Collecting layer outputs from FP32 model using %d batches...' % num_calib_batches)
include_layer = lambda name: name.endswith('_output') and (name.find('conv') != -1
                                                           or name.find('sc') != -1
                                                           or name.find('fc') != -1)
mod = mx.mod.Module(symbol=sym, context=devs, label_names=[label_name, ])
mod.bind(for_training=False,
         data_shapes=data.provide_data,
         label_shapes=data.provide_label)
mod.set_params(arg_params, aux_params)
nd_dict = collect_layer_outputs(mod, data, include_layer=include_layer, max_num_examples=num_calibrated_images,
                                logger=logger)
logger.info('Finished collecting layer outputs from FP32 model...')

logger.info('Calculating optimal thresholds for quantization...')
th_dict = mx.quantization.get_optimal_thresholds(nd_dict, logger=logger)
logger.info('Finished calculating optimal thresholds for quantization')

logger.info('Calibrating quantized model...')
calib_table_type = 'float32'
cqsym = mx.quantization.calibrate_quantized_sym(qsym, th_dict, calib_table_type)
cqsym.save('resnet_calib_%d_batches_entropy-symbol.json' % num_calib_batches)
logger.info('Finished calibrating quantized model')

data = advance_data_iter(data, max_num_calib_batches-num_calib_batches)
logger.info('Running calibrated quantized model (FP32 calibration table) for inference...')
score(cqsym, qarg_params, aux_params, data, devs, label_name, max_num_examples=num_predicted_images)
logger.info('Finished running calibrated quantized model (FP32 calibration table) for inference')
