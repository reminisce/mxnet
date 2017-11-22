import argparse
import mxnet as mx
#import os
import logging
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

fname = 'image_batch_51.nds'
image_id = 0

dirname = '/Users/jwum/Dataset/'
print('Begin loading ndarray file')
images = mx.nd.load(dirname + fname)
print('Finished loading ndarray file')

for i in range(10):
    image = images[0][i].asnumpy()
    print(image.shape)
    image = np.moveaxis(image, 0, -1)
    imgplot = plt.imshow(image)
    plt.show()
