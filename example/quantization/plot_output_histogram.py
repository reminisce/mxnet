import mxnet as mx
import matplotlib.pyplot as plt
import numpy as np

num_bins = 100
fname = 'stage3_unit9_relu3_output.nds'
fname = 'stage3_unit22_relu2_output.nds'
fname = 'stage4_unit1_relu1_output.nds'
fname = 'relu1_output.nds'
fname = 'stage3_unit1_relu1_output.nds'
fname = 'stage4_unit3_relu1_output.nds'
fname = 'stage1_unit1_conv1_output.nds'
fname = 'stage3_unit1_relu2_output.nds'
fname = 'stage1_unit1_conv2_output.nds'
image_id = 0

dirname = '/Users/jwum/Dataset/'
print('Begin loading ndarray file')
images = mx.nd.load(dirname + fname)
print('Loaded %d ndarrays' % len(images))
#tmp = (images[1] - images[0]).asnumpy()
#print(np.abs(tmp).sum())
#exit()

image = mx.nd.concat(*images, dim=0).asnumpy()
min_val = np.min(image)
max_val = np.max(image)

print('min_val=%f, max_val=%f' % (min_val, max_val))

hist, _ = np.histogram(image, bins=num_bins, range=(min_val, max_val))

bin_size = (max_val - min_val) / float(num_bins)
print(bin_size)
output_values = np.arange(start=min_val, stop=max_val, step=bin_size, dtype=np.float32)
print(output_values.size)
print(hist.size)
plt.plot(output_values[:hist.shape[0]], hist, '*')
plt.yscale('log')
title_name = '%s, min_val=%.2f, max_val=%.2f\nnum_bins=%d' % (fname[:-11], min_val, max_val, num_bins)
plt.title(title_name)
plt.show()
