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
    dist = p.astype(np.float32)
    dist += eps * is_zeros + (-eps1) * is_nonzeros
    assert (dist <= 0).sum() == 0
    #print(np.min(dist))
    return dist


num_bins = 500
num_bins = 2048
fname = 'stage4_unit1_relu1_output.nds'
fname = 'stage1_unit1_conv1_output.nds'
fname = 'stage3_unit1_relu2_output.nds'
fname = 'stage3_unit22_relu2_output.nds'
fname = 'relu1_output.nds'
fname = 'stage4_unit3_relu1_output.nds'
fname = 'stage1_unit1_conv2_output.nds'
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

#quantized_image_nd, min_val_nd, max_val_nd = mx.nd.contrib.quantize(image_nd, min_nd, max_nd)

image = image_nd.asnumpy()
min_val = min_nd.asscalar()
max_val = max_nd.asscalar()

image_hist, image_hist_edeges = np.histogram(image, bins=num_bins, range=(min_val, max_val))
#plt.plot(image_hist_edeges[0:-1], image_hist, 'o')
image_hist = smooth_distribution(image_hist)
#plt.yscale('log')
#plt.plot(image_hist_edeges[0:-1], image_hist, 'x', alpha=0.5)
#plt.show()
#exit(0)

max_range = max(abs(image_hist_edeges[0]), abs(image_hist_edeges[-1]))
thresholds = np.linspace(0, max_range+0.1, 100)
divergence = np.zeros_like(thresholds)
for i, th in enumerate(thresholds):
    quantized_image, min_nd, max_nd = mx.nd.contrib.quantize(image_nd,
                                                             mx.nd.array([-abs(th)], dtype=np.float32),
                                                             mx.nd.array([abs(th)], dtype=np.float32))
    dequantized_image = mx.nd.contrib.dequantize(quantized_image, min_nd, max_nd, out_type='float32')
    dequantized_image_hist, dequantized_image_hist_edges = np.histogram(dequantized_image.asnumpy(), bins=num_bins, range=(min_val, max_val))
    dequantized_image_hist = smooth_distribution(dequantized_image_hist)
    #dequantized_kernel = stats.gaussian_kde(dequantized_image)
    #dequantized_image_hist_smooth, dequantized_image_hist_smooth_edges =\
    #    np.histogram(dequantized_kernel(dequantized_image_hist_edges), bins=num_bins, range=(min_val, max_val))
    #dequantized_image_hist[dequantized_image_hist == 0] = 1
    #title_name = '%s, min_val=%.2f, max_val=%.2f\nnum_bins=%d' % (fname[:-11], min_val, max_val, num_bins)
    #plt.plot(image_hist_edeges[0:-1], image_hist, '.', label='Original')
    #plt.plot(dequantized_image_hist_edges[0:-1], dequantized_image_hist, '.', alpha=0.3, label='Dequantized')
    #plt.plot(dequantized_image_hist_smooth_edges[0:-1], dequantized_image_hist_smooth, '.', label='smooth')
    #plt.legend(loc='upper right')
    #plt.yscale('log')
    #plt.show()
    divergence[i] = stats.entropy(image_hist, dequantized_image_hist)
    print('%d, threshold=%f, divergence=%f' % (i, thresholds[i], divergence[i]))
    #print(divergence[i])

opt_idx = np.argmin(divergence)
print('divergence[%d]=%f, threshold[%d]=%f' % (opt_idx, divergence[opt_idx], opt_idx, thresholds[opt_idx]))
plt.plot(thresholds, divergence)
plt.show()
