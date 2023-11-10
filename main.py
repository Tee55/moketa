import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras_core as keras
import matplotlib.pyplot as plt
import time
from GAN import GAN
import imageio
import keras_core as keras

""" import torch
print(torch.cuda.get_device_name(0)) """

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

latent_dim = 300
def create_generator():
    generator = keras.Sequential(name="Generator")

    generator.add(keras.layers.Input(shape=(latent_dim, )))

    generator.add(keras.layers.Dense(8 * 8 * 512))
    generator.add(keras.layers.ReLU())
    generator.add(keras.layers.Reshape((8, 8, 512)))

    generator.add(keras.layers.Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding="same"))
    discriminator.add(keras.layers.BatchNormalization())
    generator.add(keras.layers.ReLU())

    generator.add(keras.layers.Conv2DTranspose(filters=128, kernel_size=4, strides=2, padding="same"))
    discriminator.add(keras.layers.BatchNormalization())
    generator.add(keras.layers.ReLU())

    generator.add(keras.layers.Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding="same"))
    discriminator.add(keras.layers.BatchNormalization())
    generator.add(keras.layers.ReLU())

    generator.add(keras.layers.Conv2D(filters=3, kernel_size=4, padding="same", activation='sigmoid'))
    print(generator.summary())
    return generator

def create_discriminator():
    discriminator = keras.Sequential(name="Discriminator")

    discriminator.add(keras.layers.Input(shape=(64, 64, 3)))
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

dataset = keras.utils.image_dataset_from_directory(
    "data/train", image_size=(64, 64), batch_size=32, shuffle=True
)

scale_layer = keras.layers.Rescaling(1./255)
def preprocess(images, labels):
  images = scale_layer(images)
  return images, labels

dataset_preprocess = dataset.map(preprocess)

def plot_dataset(num_samples):
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        for i in range(num_samples):
            plt.subplot(4, 4, i+1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.axis("off")
    plt.show()

plot_dataset(16)

generator = create_generator()
discriminator = create_discriminator()
noise = keras.random.normal(shape=(1, latent_dim))

writer = imageio.get_writer(
    './generated_images/{}.gif'.format(time.strftime("%Y%M%d-%H%M%S")), mode='I')


class GANMonitor(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        generated_images = self.model.generator(noise, training=False)
        generated_images = (generated_images * 255) + 255
        #output = generated_images[0].detach().cpu().numpy().astype("uint8")
        output = generated_images[0].numpy()
        img = keras.utils.array_to_img(output)
        img.save("./generated_images/generated_image_{}.png".format(epoch))
        writer.append_data(img)

gan = GAN(discriminator=discriminator,
          generator=generator, latent_dim=latent_dim)
gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    loss_fn=keras.losses.BinaryCrossentropy(),
)
gan.fit(dataset_preprocess, epochs=50, callbacks=[GANMonitor()])
writer.close()