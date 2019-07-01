import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.activations import tanh
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Flatten, Reshape, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, \
    Dense, ReLU, ZeroPadding2D, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.backend as K


class WGAN():
    def __init__(self, img_shape, latent_dim, img_helper, n_critic=5, clip_val=0.01):
        self.img_helper = img_helper
        self.img_shape = img_shape
        self.latent_dim = latent_dim

        self.n_critic = n_critic
        self.clip = clip_val


        optimizer = RMSprop(lr=5e-5)

        self.build_generator()
        self.build_discriminator(optimizer)
        self.build_wgan(optimizer)

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):
        gen_input = Input(shape=(self.latent_dim,))

        gen_model = Sequential([
            Dense(128 * 9 * 9, activation='relu', input_dim=(self.latent_dim)),
            Reshape((9, 9, 128)),
            Conv2DTranspose(128, kernel_size=4, padding='valid'),
            BatchNormalization(momentum=0.8),
            ReLU(),
            Conv2DTranspose(64, kernel_size=5, padding='valid'),
            BatchNormalization(momentum=0.8),
            ReLU(),
            Conv2DTranspose(64, kernel_size=5, padding='valid'),
            BatchNormalization(momentum=0.8),
            ReLU(),
            Conv2DTranspose(64, kernel_size=5, padding='valid'),
            BatchNormalization(momentum=0.8),
            ReLU(),
            Conv2DTranspose(self.img_shape[2], kernel_size=5, padding='valid', activation='tanh')
        ])

        gen_output = gen_model(gen_input)
        gen_model.summary()
        self.generator = Model(gen_input, gen_output)

    def build_discriminator(self, optimizer):
        dis_input = Input(shape=self.img_shape)

        dis_model = Sequential([
            Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding='same'),
            LeakyReLU(alpha=0.2),
            Dropout(0.25),
            Conv2D(32, kernel_size=3, strides=2, padding='same'),
            ZeroPadding2D(padding=((0, 1), (0, 1))),
            BatchNormalization(momentum=0.8),
            LeakyReLU(alpha=0.2),
            Dropout(0.25),
            Conv2D(64, kernel_size=3, strides=2, padding='same'),
            BatchNormalization(momentum=0.8),
            LeakyReLU(alpha=0.2),
            Dropout(0.25),
            Conv2D(128, kernel_size=3, strides=1, padding='same'),
            BatchNormalization(momentum=0.8),
            LeakyReLU(alpha=0.2),
            Dropout(0.25),
            Flatten(),
            Dense(1)
        ])

        dis_output = dis_model(dis_input)
        dis_model.summary()
        self.discriminator = Model(dis_input, dis_output)
        self.discriminator.compile(loss=self.wasserstein_loss,
                                   optimizer=optimizer,
                                   metrics=['accuracy'])
        self.discriminator.trainable = False

    def build_wgan(self, optimizer):
        real_input = Input(shape=(self.latent_dim))
        gen_output = self.generator(real_input)
        dis_output = self.discriminator(gen_output)

        self.wgan = Model(real_input, dis_output)
        self.wgan.summary()
        self.wgan.compile(loss=self.wasserstein_loss, optimizer=optimizer)

    def train(self, epochs, train_data, batch_size=128, sample_interval=100):
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))

        for epoch in range(epochs):
            for _ in range(self.n_critic):

                idx = np.random.randint(0, train_data.shape[0], batch_size)
                imgs = train_data[idx]

                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                gen_imgs = self.generator.predict(noise)

                d_loss_real = self.discriminator.train_on_batch(imgs, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                for l in self.discriminator.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip, self.clip) for w in weights]
                    l.set_weights(weights)

            g_loss = self.wgan.train_on_batch(noise, valid)

            print("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss))

            if epoch % sample_interval == 0:
                self.save_images(epoch)

    def save_images(self, epoch):
        generated = self.predict_noise(25)
        generated = 0.5 * generated + 0.5
        self.img_helper.save_image(generated, epoch, "generated/")

    def predict_noise(self, size):
        noise = np.random.normal(0, 1, (size, self.latent_dim))
        return self.generator.predict(noise)

    def plot_loss(self, history):
        hist = pd.DataFrame(history)
        plt.figure(figsize=(20, 5))
        for col in hist.columns:
            plt.plot(hist[col], label=col)
        plt.legend()
        plt.ylabel("loss")
        plt.xlabel("epochs")
        plt.show()

if __name__ == "__main__":
    from ..GAN.image_helper import ImageHelper

    (X, _), (_, _) = mnist.load_data()
    X_train = X / 127.5 - 1.
    X_train = np.expand_dims(X_train, axis=3)
    wgan = WGAN(img_shape=X_train[0].shape, latent_dim=100, img_helper=ImageHelper())
    wgan.train(epochs=4000, train_data=X_train, batch_size=32, sample_interval=50)
