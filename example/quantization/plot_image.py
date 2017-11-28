import argparse
import mxnet as mx
#import os
import logging
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

fname = 'image_batch_200.nds'
image_id = 0

dirname = '/Users/jwum/Dataset/'
print('Begin loading ndarray file')
images = mx.nd.load(dirname + fname)[0]
print('Finished loading ndarray file')

for i in range(images.shape[0]):
    image = images[i].asnumpy()
    print(image)
    image = np.moveaxis(image, 0, -1)
    imgplot = plt.imshow(image)
    plt.show()
