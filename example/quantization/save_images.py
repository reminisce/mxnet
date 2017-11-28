import argparse
import mxnet as mx
import logging

parser = argparse.ArgumentParser(description='score a model on a dataset')
parser.add_argument('--rgb-mean', type=str, default='128,128,128')
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--image-shape', type=str, default='3,299,299')
parser.add_argument('--data-nthreads', type=int, default=32,
                    help='number of threads for data decoding')
parser.add_argument('--batch-id', type=int, default=0, help='batch id to be saved to file')
args = parser.parse_args()

data_nthreads = args.data_nthreads
image_shape = args.image_shape
# model = args.model
rgb_mean = args.rgb_mean
batch_size = args.batch_size
label_name = 'softmax_label'

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

mean_img = None

# create data iterator
data_shape = tuple([int(i) for i in image_shape.split(',')])
if mean_img is not None:
    mean_args = {'mean_img': mean_img}
elif rgb_mean is not None:
    rgb_mean = [float(i) for i in rgb_mean.split(',')]
    mean_args = {'mean_r': rgb_mean[0], 'mean_g': rgb_mean[1], 'mean_b': rgb_mean[2]}

# data_filename = 'val-5k-256.rec'
# data_filename = 'val_256_q90.rec'
data_filename = 'val_480_q95.rec'
data_dirname = 'data'
data_val = data_dirname + '/' + data_filename
batch_id = args.batch_id
assert batch_id >= 0

data_iter = mx.io.ImageRecordIter(path_imgrec=data_val,
                                  label_width=1,
                                  preprocess_threads=data_nthreads,
                                  batch_size=batch_size,
                                  data_shape=data_shape,
                                  resize=299,
                                  label_name=label_name,
                                  rand_crop=False,
                                  rand_mirror=False,
                                  scale=1./128.,
                                  **mean_args)


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


batch_id = 0
while batch_id >= 0:
    data_iter.reset()
    batch_id = int(input("Enter batch id: "))
    if batch_id < 0:
        exit(0)
    data_iter = advance_data_iter(data_iter, batch_id)
    batch = data_iter.next()
    data_iter.reset()
    images = batch.data[0]
    fname = 'image_batch_%d.nds' % batch_id
    print('Saving images into file %s' % fname)
    mx.nd.save(fname, images)
