import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Flatten, Reshape, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.backend as K


class WGAN():
    def __init__(self, img_shape, latent_dim, img_helper):
        self.img_helper = img_helper
        self.img_shape = img_shape
        self.hidden_dim = latent_dim

        optimizer = RMSprop(lr=5e-5)

        self.build_generator()
        self.build_discriminator(optimizer)
        self.build_gan(optimizer)

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)