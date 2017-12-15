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


num_bins = 513
num_bins = 1025
num_bins = 4001
num_bins = 1001
num_bins = 4097
num_bins = 2001
num_bins = 1001
num_bins = 3001
num_bins = 40001
num_bins = 6001
num_bins = 3001
num_bins = 2049
num_bins = 8001
num_quantized_bins = 255
assert num_bins % 2 == 1

fname = 'stage3_unit1_relu2_output.nds'
fname = 'stage3_unit22_relu2_output.nds'
fname = 'stage3_unit1_relu1_output.nds'
fname = 'stage4_unit1_relu1_output.nds'
fname = 'relu1_output.nds'
fname = 'stage3_unit9_relu3_output.nds'
fname = 'stage1_unit1_conv2_output.nds'
fname = 'stage1_unit1_conv1_output.nds'
fname = 'fc1_output.nds'
fname = 'stage4_unit3_relu1_output.nds'
image_id = 0

dirname = '/Users/jwum/Dataset/'
print('Begin loading ndarray file')
images = mx.nd.load(dirname + fname)
print('Loaded %d ndarrays' % len(images))

image_nd = mx.nd.concat(*images, dim=0)
min_nd = mx.nd.min(image_nd)
max_nd = mx.nd.max(image_nd)

quantized_image_nd, min_val_nd, max_val_nd = mx.nd.contrib.quantize(image_nd, min_nd, max_nd)
quantized_image = quantized_image_nd.asnumpy()

image = image_nd.asnumpy()
min_val = min_nd.asscalar()
max_val = max_nd.asscalar()
th = max(abs(min_val), abs(max_val))
#image = np.random.uniform(low=-th, high=th, size=image.shape)
#image = np.random.normal(0.0, 3.0, size=image.shape)
#min_val = np.min(image)
#max_val = np.max(image)
#image = np.abs(image)
print('min_val=%f, max_val=%f' % (min_val, max_val))


image_hist, image_hist_edeges = np.histogram(image, bins=num_bins, range=(-th, th))
zero_bin_idx = num_bins / 2
num_half_quantized_bins = num_quantized_bins / 2
assert np.allclose(image_hist_edeges[zero_bin_idx] + image_hist_edeges[zero_bin_idx+1], 0, rtol=1e-5, atol=1e-7)

thresholds = np.zeros(num_bins/2 + 1 - num_quantized_bins/2)
divergence = np.zeros_like(thresholds)
quantized_bins = np.zeros(num_quantized_bins, dtype=np.int32)
for i in range(num_quantized_bins/2, num_bins/2+1):  # i means the number of bins on half axis excluding the zero bin
    p_bin_idx_start = zero_bin_idx - i
    p_bin_idx_stop = zero_bin_idx + i + 1
    thresholds[i-num_half_quantized_bins] = image_hist_edeges[p_bin_idx_stop]
    # sliced_image_hist is used to generate candidate distribution q
    sliced_image_hist = image_hist[p_bin_idx_start:p_bin_idx_stop]

    # generate reference distribution p
    p = sliced_image_hist.copy()
    assert p.size % 2 == 1
    assert p.size >= num_quantized_bins
    # put left outlier count in p[0]
    left_outlier_count = np.sum(image_hist[0:p_bin_idx_start])
    p[0] += left_outlier_count
    # put right outlier count in p[-1]
    right_outlier_count = np.sum(image_hist[p_bin_idx_stop:])
    p[-1] += right_outlier_count
    # is_nonzeros[k] indicates whether image_hist[k] is nonzero
    is_nonzeros = (sliced_image_hist != 0).astype(np.int32)

    # calculate how many bins should be merged to generate quantized distribution q
    num_merged_bins = p.size / num_quantized_bins
    # merge image_hist into num_quantized_bins bins
    for j in range(num_quantized_bins):
        start = j * num_merged_bins
        stop = start + num_merged_bins
        quantized_bins[j] = sliced_image_hist[start:stop].sum()
    quantized_bins[-1] += sliced_image_hist[num_quantized_bins*num_merged_bins:].sum()
    # expand quantized_bins into p.size bins
    q = np.zeros(p.size, dtype=np.float32)
    for j in range(num_quantized_bins):
        start = j * num_merged_bins
        if j == num_quantized_bins - 1:
            stop = -1
        else:
            stop = start + num_merged_bins
        norm = is_nonzeros[start:stop].sum()
        if norm != 0:
            q[start:stop] = float(quantized_bins[j]) / float(norm)
    q[sliced_image_hist == 0] = 0
    p = smooth_distribution(p)
    q = smooth_distribution(q)
    divergence[i-num_half_quantized_bins] = stats.entropy(p, q)
    quantized_bins[:] = 0

min_kl_divergence_idx = np.argmin(divergence)
print('min kl divergence: divergence[%d]=%f' % (min_kl_divergence_idx, divergence[min_kl_divergence_idx]))
print('threshold: %f' % thresholds[min_kl_divergence_idx])

title_name = '%s, min_val=%.2f, max_val=%.2f\nnum_bins=%d' % (fname[:-11], min_val, max_val, num_bins)
plt.subplot(2, 1, 1)
plt.title(title_name)
plt.plot(thresholds, divergence, '.', label='kl divergence')
plt.legend(loc='best')

plt.subplot(2, 1, 2)
bin_width = image_hist_edeges[1] - image_hist_edeges[0]
plt.plot(image_hist_edeges[:image_hist.shape[0]]+bin_width/2, image_hist, '.', label='layer output')
plt.axvline(x=thresholds[min_kl_divergence_idx], color='r', label='threshold')
plt.axvline(x=-thresholds[min_kl_divergence_idx], color='r', label='threshold')
plt.legend(loc='best')
plt.yscale('log')
plt.show()
