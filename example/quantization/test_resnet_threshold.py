import argparse
from common import modelzoo
import mxnet as mx
import time
import os
import logging
from mxnet.quantization import *


parser = argparse.ArgumentParser(description='score a model on a dataset')
parser.add_argument('--gpus', type=str, default='0')
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--rgb-mean', type=str, default='0,0,0')
parser.add_argument('--data-val', type=str, required=True)
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
max_num_calib_batches = 50
num_calib_batches = 5
assert num_calib_batches <= max_num_calib_batches

num_infer_batches = 500
num_infer_image_offset = batch_size * max_num_calib_batches
num_predicted_images = batch_size * num_infer_batches
num_calibrated_images = batch_size * num_calib_batches

data_nthreads = args.data_nthreads
data_val = args.data_val
gpus = args.gpus
image_shape = args.image_shape
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


#data_filename = 'val-5k-256.rec'
data_filename = 'val_256_q90.rec'
data_dirname = 'data'
data_val = data_dirname + '/' + data_filename
url = 'http://data.mxnet.io/data/' + data_filename


def download_data():
    return mx.test_utils.download(url=url, fname=data_filename, dirname=data_dirname, overwrite=False)


print('Downloading validation dataset from %s' % url)
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


# create module
if gpus == '':
    devs = mx.cpu()
else:
    devs = [mx.gpu(int(i)) for i in gpus.split(',')]


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

while True:
    stop = input('Please enter a number 0=stop, 1=continue: ')
    if stop == 0:
        break
    data.reset()
    data = advance_data_iter(data, num_infer_image_offset/batch_size)
    metrics = [mx.metric.create('acc'),
               mx.metric.create('top_k_accuracy', top_k=5)]
    if not isinstance(metrics, list):
        metrics = [metrics, ]
    mod = mx.mod.Module.load('quantized-resnet', 0, context=devs, label_names=[label_name,])

    mod.bind(for_training=False,
             data_shapes=data.provide_data,
             label_shapes=data.provide_label)
    #mod.set_params(arg_params, aux_params)

    tic = time.time()
    num = 0
    for batch in data:
        mod.forward(batch, is_train=False)
        for m in metrics:
            mod.update_metric(m, batch.label)
        num += batch_size
        if num >= num_predicted_images:
            break

    speed = num / (time.time() - tic)

    logging.info('Finished inference with %d images' % num)
    logging.info('Finished with %f images per second', speed)
    for m in metrics:
        logging.info(m.get())
    print('Finished running calibrated quantized model (FP32 calibration table) for inference')
    data.reset()
    print('\n')
#################################################################################################
