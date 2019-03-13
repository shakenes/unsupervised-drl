from __future__ import print_function
import keras
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=None)
args = parser.parse_args()

save_dir = os.path.join(os.getcwd(), 'saved_models')

model = keras.models.load_model(os.path.join(save_dir, args.model))

for x in [2, 3]:   # [2, 3, 4]
    W = model.layers[x].get_weights()[0][:, :, 0, :]  # filter width, filter height, channel, filter number

    ncols = 8
    nrows = W.shape[2]//ncols
    f, axarr = plt.subplots(nrows=nrows, ncols=ncols)

    dim = W.shape
    print(dim)

    for i in range(nrows):
        for k in range(ncols):
            x = W[:, :, ncols*i+k]
            x -= x.mean()
            x /= (x.std() + 1e-5)
            x *= 0.1

            # clip to [0, 1]
            x += 0.5
            x = np.clip(x, 0, 1)

            x *= 255
            x = np.clip(x, 0, 255).astype('uint8')

            axarr[i, k].imshow(x, cmap=plt.get_cmap('Greys'))
            axarr[i, k].axis('off')

    # plt.suptitle("Filter kernels of {} convolutional layer".format(n))
    plt.show()
    f.savefig('test.eps', format='eps', bbox_inches='tight', pad_inches=0)