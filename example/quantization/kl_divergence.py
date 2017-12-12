import mxnet as mx
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def smooth_distribution(p, eps=0.0001):
    is_zeros = (p == 0).astype(np.float32)
    is_nonzeros = (p != 0).astype(np.float32)
    n_zeros = is_zeros.sum()
    n_nonzeros = p.size - n_zeros
    eps1 = eps * float(n_zeros) / float(n_nonzeros)
    assert eps1 < 1.0, 'n_zeros=%d, n_nonzeros=%d, eps1=%f' % (n_zeros, n_nonzeros, eps1)
    hist = p.astype(np.float32)
    hist += eps * is_zeros + (-eps1) * is_nonzeros
    assert (hist <= 0).sum() == 0
    return hist


num_bins = 512
num_bins = 1024
num_bins = 4096
num_bins = 4000
num_bins = 1000
num_bins = 40000
num_bins = 2048
fname = 'stage4_unit1_relu1_output.nds'
fname = 'stage1_unit1_conv1_output.nds'
fname = 'stage3_unit1_relu2_output.nds'
fname = 'stage3_unit22_relu2_output.nds'
fname = 'stage1_unit1_conv2_output.nds'
fname = 'relu1_output.nds'
fname = 'stage4_unit3_relu1_output.nds'
fname = 'stage3_unit9_relu3_output.nds'
fname = 'stage3_unit1_relu1_output.nds'
image_id = 0

dirname = '/Users/jwum/Dataset/'
print('Begin loading ndarray file')
images = mx.nd.load(dirname + fname)
print('Loaded %d ndarrays' % len(images))
#tmp = (images[1] - images[0]).asnumpy()
#print(np.abs(tmp).sum())
#exit()

image_nd = mx.nd.concat(*images, dim=0)
min_nd = mx.nd.min(image_nd)
max_nd = mx.nd.max(image_nd)
print('min_val=%f, max_val=%f' % (min_nd.asscalar(), max_nd.asscalar()))
#min_nd[:] = -0.5
#max_nd[:] = 0.5

quantized_image_nd, min_val_nd, max_val_nd = mx.nd.contrib.quantize(image_nd, min_nd, max_nd)
quantized_image = quantized_image_nd.asnumpy()
#print(quantized_image)
#print(min_val_nd)
#print(max_val_nd)

image = image_nd.asnumpy()
min_val = min_nd.asscalar()
max_val = max_nd.asscalar()
#image = np.random.uniform(low=min_val, high=max_val, size=image.shape)
#image = np.random.normal(5.0, 1.0, size=image.shape)
#image = np.abs(image)

image_hist, image_hist_edeges = np.histogram(image, bins=num_bins)
bin_width = image_hist_edeges[1] - image_hist_edeges[0]
p = None
divergence = np.zeros(num_bins-128, np.float32)
bins128 = np.zeros(128, dtype=np.int32)
for i in range(128, num_bins):
    p = image_hist[0:i].copy()
    outlier_count = np.sum(image_hist[i:])
    p[i-1] += outlier_count
    is_nonzeros = (image_hist[0:i] != 0).astype(np.int32)  # is_nonzeros[k] indicates whether image_hist[k] is nonzero
    #is_nonzeros = (p[0:i] != 0).astype(np.int32)  # is_nonzeros[k] indicates whether p[k] is zero
    num_merged_bins = i / 128
    # merge image_hist into 128 bins
    for j in range(128):
        start = j * num_merged_bins
        if j == 127:
            stop = i
        else:
            stop = start + num_merged_bins
        bins128[j] += image_hist[start:stop].sum()
        #bins128[j] += p[start:stop].sum()
    #print(bins128)
    # expand bins128 into i bins
    #q = np.zeros(i, dtype=np.int32)
    q = np.zeros(i, dtype=np.float32)
    for j in range(128):
        start = j * num_merged_bins
        if j == 127:
            stop = i
        else:
            stop = start + num_merged_bins
        norm = is_nonzeros[start:stop].sum()
        if norm != 0:
            q[start:stop] = float(bins128[j]) / float(norm)
    #q[image_hist[0:i] == 0] = 0
    #q[p == 0] = 0
    #p[q == 0] = 0
    #if not q[p != 0].all():
    #    for k in range(p.size):
    #        if p[k] != 0 and q[k] == 0:
    #            raise ValueError('When i=%d, p[%d]=%d, q[%d]=%f' % (i, k, p[k], k, q[k]))
    q = smooth_distribution(q)
    divergence[i-128] = stats.entropy(p, q)
    bins128[:] = 0

min_kl_divergence_idx = np.argmin(divergence)
print('Min kl divergence: divergence[%d]=%f' % (min_kl_divergence_idx, divergence[min_kl_divergence_idx]))
#threshold = (min_kl_divergence_idx + 128 + 0.5) * bin_width
threshold = image_hist_edeges[min_kl_divergence_idx+128]
print('threshold: %f' % threshold)

title_name = '%s, min_val=%.2f, max_val=%.2f\nnum_bins=%d' % (fname[:-11], min_val, max_val, num_bins)
#title_name = '%s, min_val=%.2f, max_val=%.2f\nnum_bins=%d' % ('normal dist, mean=5, sigma=1', min_val, max_val, num_bins)
plt.subplot(2, 1, 1)
plt.title(title_name)
plt.plot(image_hist_edeges[128:num_bins], divergence, label='kl divergence')
plt.legend(loc='upper left')

plt.subplot(2, 1, 2)
plt.plot(image_hist_edeges[:image_hist.shape[0]], image_hist, '.', label='layer output')
plt.axvline(x=threshold, color='r', label='threshold')
plt.legend(loc='upper right')
plt.yscale('log')
plt.show()
