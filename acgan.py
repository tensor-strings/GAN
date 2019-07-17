import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from image_helper import ImageHelper

from tensorflow.keras.activations import tanh
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Flatten, Reshape, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, \
    Dense, ReLU, ZeroPadding2D, Dropout, multiply, Embedding
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.backend as K


class ACGAN(object):
    def __init__(self, img_shape, latent_dim, num_classes, optimizer=Adam(lr=1e-4), dataset='', img_helper=ImageHelper()):
        self.img_helper = img_helper
        self.img_shape = img_shape
        self.latent_dim = latent_dim

        self.num_classes = num_classes

        self.optimizer = optimizer

        self.dataset = dataset

    def boundary_loss(self, y_true, y_pred):
        return 0.5 * K.mean((K.log(y_pred) - K.log(1 - y_pred)) ** 2)

    def build_generator(self):
        gen_input_noise = Input(shape=(self.latent_dim,))
        gen_input_label = Input(shape=(1,), dtype='int32')
        gen_input_label_embedding = Flatten()(Embedding(self.num_classes, 100, input_length=1)(gen_input_label))
        gen_input = multiply([gen_input_noise, gen_input_label_embedding])

        gen_model = Sequential([
            Dense(128 * 7 * 7, activation='relu', input_dim=(self.latent_dim)),
            Reshape((7, 7, 128)),
            Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'),
            BatchNormalization(momentum=0.8),
            ReLU(),
            Conv2DTranspose(64, kernel_size=3, strides=2, padding='same'),
            BatchNormalization(momentum=0.8),
            ReLU(),
            Conv2DTranspose(self.img_shape[2], kernel_size=3, padding='same', activation='tanh')
        ])

        gen_output = gen_model(gen_input)
        gen_model.summary()
        self.generator = Model([gen_input_noise, gen_input_label], gen_output)

    def build_discriminator(self):
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
        ])

        dis_features_output = dis_model(dis_input)

        validity = Dense(1, activation='sigmoid')(dis_features_output)
        label = Dense(self.num_classes, activation='softmax')(dis_features_output)

        dis_model.summary()
        self.discriminator = Model(dis_input, [validity, label])
        self.discriminator.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'],
                                   optimizer=self.optimizer,
                                   metrics=['accuracy'])
        self.discriminator.trainable = False

    def build_acgan(self):
        real_input = Input(shape=(self.latent_dim))
        label_input = Input(shape=(1,), dtype='int32')

        gen_output = self.generator([real_input, label_input])
        dis_output = self.discriminator([gen_output, label_input])

        self.acgan = Model([real_input, label_input], dis_output)
        self.acgan.summary()
        self.acgan.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=self.optimizer)

    def train(self, epochs, train_data, batch_size, sample_interval=100):
        self.build_generator()
        self.build_discriminator()
        self.build_acgan()

        X_train, y_train = train_data
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        history = []
        for epoch in range(epochs):
            batch_indices = np.random.randint(0, X_train.shape[0], batch_size)
            batch = X_train[batch_indices]
            batch_labels = y_train[batch_indices]
            generated, gen_sampled_labels = self.predict_noise(batch_size)
            loss_real = self.discriminator.train_on_batch(batch, [real, batch_labels])
            loss_fake = self.discriminator.train_on_batch(generated, [fake, gen_sampled_labels])
            dis_loss = 0.5 * np.add(loss_real, loss_fake)

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            sampled_labels = np.random.randint(0, 10, batch_size)

            gen_loss = self.acgan.train_on_batch([noise, sampled_labels], [real, sampled_labels])

            print("Epoch " + str(epoch) + ", Dis loss: " + str(dis_loss[0]) + ", Gen loss: " + str(gen_loss))

            history.append({'D': dis_loss[0], 'G': gen_loss})

            if epoch % sample_interval == 0:
                self.save_images(epoch)

        self.plot_loss(history)
        self.img_helper.makegif("generated/", self.dataset)

    def save_images(self, epoch):
        generated, labels = self.predict_noise(25)
        generated = 0.5 * generated + 0.5
        self.img_helper.save_image(generated, epoch, "generated/acgan/", self.dataset)

    def predict_noise(self, size):
        noise = np.random.normal(0, 1, (size, self.latent_dim))
        sampled_labels = np.random.randint(0, 10, size)
        return self.generator.predict([noise, sampled_labels]), sampled_labels

    def plot_loss(self, history):
        hist = pd.DataFrame(history)
        plt.figure(figsize=(20, 5))
        for col in hist.columns:
            plt.plot(hist[col], label=col)
        plt.legend()
        plt.ylabel("loss")
        plt.xlabel("epochs")
        plt.show()


class WACGAN(ACGAN):
    def __init__(self, img_shape, latent_dim, num_classes, optimizer=RMSprop(lr=5e-6),
                 n_critic=5, clip_val=0.01, dataset=''):
        super().__init__(img_shape, latent_dim, num_classes, dataset=dataset)

        self.n_critic = n_critic
        self.clip = clip_val
        self.optimizer = optimizer

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):
        gen_input_noise = Input(shape=(self.latent_dim,))
        gen_input_label = Input(shape=(1,), dtype='int32')
        gen_input_label_embedding = Flatten()(Embedding(self.num_classes, 100, input_length=1)(gen_input_label))
        gen_input = multiply([gen_input_noise, gen_input_label_embedding])

        gen_model = Sequential([
            Dense(128 * 7 * 7, activation='relu', input_dim=(self.latent_dim)),
            Reshape((7, 7, 128)),
            Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'),
            BatchNormalization(momentum=0.8),
            ReLU(),
            Conv2DTranspose(64, kernel_size=3, strides=2, padding='same'),
            BatchNormalization(momentum=0.8),
            ReLU(),
            Conv2DTranspose(self.img_shape[2], kernel_size=3, padding='same', activation='tanh')
        ])

        gen_output = gen_model(gen_input)
        gen_model.summary()
        self.generator = Model([gen_input_noise, gen_input_label], gen_output)

    def build_discriminator(self):
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
        ])

        dis_features_output = dis_model(dis_input)

        validity = Dense(1, activation='sigmoid')(dis_features_output)
        label = Dense(self.num_classes, activation='softmax')(dis_features_output)

        dis_model.summary()
        self.discriminator = Model(dis_input, [validity, label])
        self.discriminator.compile(loss=[self.wasserstein_loss, 'sparse_categorical_crossentropy'],
                                   optimizer=self.optimizer,
                                   metrics=['accuracy'])
        self.discriminator.trainable = False

    def build_acgan(self):
        real_input = Input(shape=(self.latent_dim))
        label_input = Input(shape=(1,), dtype='int32')

        gen_output = self.generator([real_input, label_input])
        dis_output = self.discriminator([gen_output, label_input])

        self.acgan = Model([real_input, label_input], dis_output)
        self.acgan.summary()
        self.acgan.compile(loss=[self.wasserstein_loss, 'sparse_categorical_crossentropy'], optimizer=self.optimizer)

    def train(self, epochs, train_data, batch_size=128, sample_interval=100):
        self.build_generator()
        self.build_discriminator()
        self.build_acgan()

        X_train, y_train = train_data
        real = np.ones((batch_size, 1))
        fake = -np.ones((batch_size, 1))
        history = []

        for epoch in range(epochs):
            for _ in range(self.n_critic):

                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]
                batch_labels = y_train[idx]

                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                generated, gen_sampled_labels = self.predict_noise(batch_size)

                d_loss_real = self.discriminator.train_on_batch(imgs, [real, batch_labels])
                d_loss_fake = self.discriminator.train_on_batch(generated, [fake, gen_sampled_labels])
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                for l in self.discriminator.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip, self.clip) for w in weights]
                    l.set_weights(weights)

            sampled_labels = np.random.randint(0, 10, batch_size)

            g_loss = self.acgan.train_on_batch([noise, sampled_labels], [real, sampled_labels])

            print("Epoch " + str(epoch) + ", Dis loss: " + str(d_loss[0]-1) + ", Gen loss: " + str(g_loss[0]-1))

            if epoch % sample_interval == 0:
                self.save_images(epoch)

        self.img_helper.makegif("generated/", self.dataset)


if __name__ == "__main__":
    from image_helper import ImageHelper

    (X, y), (_, _) = mnist.load_data()
    X_train = X / 127.5 - 1.
    X_train = np.expand_dims(X_train, axis=3)
    y_train = y.reshape(-1,1)
    acgan = WACGAN(img_shape=X_train[0].shape, latent_dim=100, num_classes=10, dataset='mnist')
    acgan.train(epochs=40000, train_data=[X_train,y_train], batch_size=128, sample_interval=100)