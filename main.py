import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras_core as keras
from keras_core.datasets import mnist
from keras_core.layers import Input, Dense, Reshape, Flatten, Dropout
from keras_core.layers import BatchNormalization, Activation, ZeroPadding2D
from keras_core.layers import LeakyReLU
from keras_core.layers import UpSampling2D, Conv2D
from keras_core.models import Sequential, Model
from keras_core.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np
import cv2


class DCGAN():
    def __init__(self):
        # Input shape
        self.img_width = 64
        self.img_height = 64
        self.channels = 3
        self.img_shape = (self.img_width, self.img_height, self.channels)
        self.latent_dim = 200

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        generator = keras.Sequential(name="Generator")

        generator.add(keras.layers.Input(shape=(self.latent_dim, )))

        generator.add(keras.layers.Dense(8 * 8 * 512))
        generator.add(keras.layers.ReLU())
        generator.add(keras.layers.Reshape((8, 8, 512)))

        generator.add(keras.layers.Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding="same"))
        generator.add(keras.layers.BatchNormalization())
        generator.add(keras.layers.ReLU())

        generator.add(keras.layers.Conv2DTranspose(filters=128, kernel_size=4, strides=2, padding="same"))
        generator.add(keras.layers.BatchNormalization())
        generator.add(keras.layers.ReLU())

        generator.add(keras.layers.Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding="same"))
        generator.add(keras.layers.BatchNormalization())
        generator.add(keras.layers.ReLU())

        generator.add(keras.layers.Conv2D(filters=self.channels, kernel_size=4, padding="same", activation='sigmoid'))
        print(generator.summary())

        return generator

    def build_discriminator(self):

        discriminator = keras.Sequential(name="Discriminator")

        discriminator.add(keras.layers.Input(shape=(self.img_width, self.img_height, self.channels)))
        #discriminator.add(keras.layers.Rescaling(1./255))

        discriminator.add(keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, padding="same"))
        discriminator.add(keras.layers.BatchNormalization())
        discriminator.add(keras.layers.LeakyReLU())

        discriminator.add(keras.layers.Conv2D(filters=128, kernel_size=4, strides=2, padding="same"))
        discriminator.add(keras.layers.BatchNormalization())
        discriminator.add(keras.layers.LeakyReLU())

        discriminator.add(keras.layers.Conv2D(filters=256, kernel_size=4, strides=2, padding="same"))
        discriminator.add(keras.layers.BatchNormalization())
        discriminator.add(keras.layers.LeakyReLU())

        discriminator.add(keras.layers.Flatten())
        discriminator.add(keras.layers.Dense(units=1, activation="sigmoid"))
        print(discriminator.summary())
        
        return discriminator

    def train(self, epochs, batch_size=128, save_interval=50):

        dir_path = 'data/train/faces'

        X_train = []
        for filename in os.listdir(dir_path):
            image = keras.utils.load_img(os.path.join(dir_path, filename), target_size=(self.img_height, self.img_width))
            image = keras.utils.img_to_array(image)
            X_train.append(image)

        X_train = np.array(X_train)

         # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
                  (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        """ for i in range(3):
            img = keras.utils.array_to_img(gen_imgs[i])
            img.save("generated_img_%03d_%d.png" % (epoch, i)) """

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("generated_images/mnist_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs=4000, batch_size=32, save_interval=50)
