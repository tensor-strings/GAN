# from https://github.com/NMZivkovic/gan-dcgan.git

import os
import numpy as np
import imageio
import matplotlib.pyplot as plt


class ImageHelper(object):
    def save_image(self, generated, epoch, directory, dataset):
        fig, axs = plt.subplots(5, 5)
        count = 0
        for i in range(5):
            for j in range(5):
                axs[i, j].imshow(generated[count, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                count += 1
        fig.savefig('{}/{}_{}.png'.format(directory, dataset, epoch))
        plt.close()

    def makegif(self, directory, dataset):
        filenames = np.sort(os.listdir(directory))
        filenames = [fnm for fnm in filenames if ".png" in fnm]

        with imageio.get_writer(directory + '/{}_image.gif'.format(dataset), mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(directory + filename)
                writer.append_data(image)