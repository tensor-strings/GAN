import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.activations import tanh
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Flatten, Reshape, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, \
    Dense, ReLU, ZeroPadding2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.backend as K


class BSGAN():

    def __init__(self, img_shape, latent_dim, img_helper, dataset=''):
        self.img_helper = img_helper
        self.img_shape = img_shape
        self.latent_dim = latent_dim

        self.dataset = dataset

        optimizer = Adam(1e-4, 0.5)

        self.build_generator()
        self.build_discriminator(optimizer)
        self.build_bsgan(optimizer)

    def boundary_loss(self, y_true, y_pred):
        return 0.5 * K.mean((K.log(y_pred) - K.log(1 - y_pred))**2)

    def build_generator(self):
        gen_input = Input(shape=(self.latent_dim,))

        gen_model = Sequential([
            Dense(256, input_dim=(self.latent_dim)),
            LeakyReLU(alpha=0.2),
            BatchNormalization(momentum=0.8),
            Dense(512),
            LeakyReLU(alpha=0.2),
            BatchNormalization(momentum=0.8),
            Dense(1024),
            LeakyReLU(alpha=0.2),
            BatchNormalization(momentum=0.8),
            Dense(np.prod(self.img_shape), activation='tanh'),
            Reshape(self.img_shape)
        ])

        gen_output = gen_model(gen_input)
        gen_model.summary()
        self.generator = Model(gen_input, gen_output)

    def build_discriminator(self, optimizer):
        dis_input = Input(shape=self.img_shape)

        dis_model = Sequential([
            Flatten(input_shape=self.img_shape),
            Dense(512),
            LeakyReLU(alpha=0.2),
            Dense(256),
            LeakyReLU(alpha=0.2),
            Dense(1, activation='sigmoid')
        ])

        dis_output = dis_model(dis_input)
        self.discriminator = Model(dis_input, dis_output)
        self.discriminator.summary()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])
        self.discriminator.trainable = False

    def build_bsgan(self, optimizer):
        real_input = Input(shape=(self.latent_dim))
        gen_output = self.generator(real_input)
        dis_output = self.discriminator(gen_output)

        self.bsgan = Model(real_input, dis_output)
        self.bsgan.summary()
        self.bsgan.compile(loss=self.boundary_loss, optimizer=optimizer)

    def train(self, epochs, train_data, batch_size):
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        history = []
        for epoch in range(epochs):
            batch_indices = np.random.randint(0, train_data.shape[0], batch_size)
            batch = train_data[batch_indices]
            generated = self.predict_noise(batch_size)
            loss_real = self.discriminator.train_on_batch(batch, real)
            loss_fake = self.discriminator.train_on_batch(generated, fake)
            dis_loss = 0.5 * np.add(loss_real, loss_fake)

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_loss = self.bsgan.train_on_batch(noise, real)

            print("Epoch " + str(epoch) + ", Dis loss: " + str(dis_loss[0]) + ", Gen loss: " + str(gen_loss))

            history.append({'D': dis_loss[0], 'G': gen_loss})

            if epoch % 100 == 0:
                self.save_images(epoch)

        self.plot_loss(history)
        self.img_helper.makegif("generated/", self.dataset)

    def save_images(self, epoch):
        generated = self.predict_noise(25)
        generated = 0.5 * generated + 0.5
        self.img_helper.save_image(generated, epoch, "generated/bsgan/", self.dataset)

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
    from keras.datasets import mnist

    from image_helper import ImageHelper

    (X, _), (_, _) = mnist.load_data()
    X_train = X / 127.5 - 1.
    X_train = np.expand_dims(X_train, axis=3)

    image_helper = ImageHelper()
    bsgan = BSGAN(X_train[0].shape, 100, image_helper)
    bsgan.train(30000, X_train, batch_size=256)