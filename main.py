import os
os.environ["KERAS_BACKEND"] = "torch"
import keras_core as keras
import matplotlib.pyplot as plt
import time
from GAN import GAN
import imageio
import keras_core as keras
import tensorflow as tf
import torch
from torchvision.utils import save_image
from torchvision import transforms
import numpy as np
import cv2

dataset = keras.utils.image_dataset_from_directory(
    "face_images", label_mode=None, image_size=(64, 64), batch_size=32
)

def plot_dataset(num_samples):
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        for i in range(num_samples):
            plt.subplot(4, 4, i+1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.axis("off")
    plt.show()

#plot_dataset(16)
'''
def process(image, label):
    image = tf.cast(image/255. ,tf.float32)
    return image, label

dataset = dataset.map(process)
'''

# normalize the images
dataset = dataset.map(lambda x: (x - 127.5) / 127.5)

latent_dim = 100
def create_generator():
    model = keras.Sequential(name="Generator")
    model.add(keras.layers.Dense(8 * 8 * 512, input_shape=(latent_dim, )))
    model.add(keras.layers.ReLU())
    model.add(keras.layers.Reshape((8, 8, 512)))

    model.add(keras.layers.Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding="same"))
    model.add(keras.layers.ReLU())

    model.add(keras.layers.Conv2DTranspose(filters=128, kernel_size=4, strides=2, padding="same"))
    model.add(keras.layers.ReLU())

    model.add(keras.layers.Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding="same"))
    model.add(keras.layers.ReLU())

    model.add(keras.layers.Conv2D(filters=3, kernel_size=4, padding="same", activation='tanh'))
    print(model.summary())
    return model

def create_discriminator():
    model = keras.Sequential(name="Discriminator")
    model.add(keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, padding="same", input_shape=(64, 64, 3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha = 0.2))

    model.add(keras.layers.Conv2D(filters=128, kernel_size=4, strides=2, padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha = 0.2))

    model.add(keras.layers.Conv2D(filters=128, kernel_size=4, strides=2, padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha = 0.2))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(rate=0.3))
    model.add(keras.layers.Dense(units=1, activation="sigmoid"))
    print(model.summary())
    return model

generator = create_generator()
discriminator = create_discriminator()
toPILImage = transforms.ToPILImage()
noise = keras.random.normal(shape=(1, latent_dim), mean=0.0, stddev=0.02)

def plot_generator():
    generated_images = generator(noise, training=False)
    generated_images *= (generated_images * 127.5) + 127.5
    output = generated_images[0].cpu().detach().numpy()
    img = keras.utils.array_to_img(output)
    img.save("./generated_images/generated_image_{}.png".format(0))

plot_generator()

gan = GAN(discriminator=discriminator,
          generator=generator, latent_dim=latent_dim)
gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    loss_fn=keras.losses.BinaryCrossentropy(),
)

writer = imageio.get_writer(
    './generated_images/{}.gif'.format(time.strftime("%Y%M%d-%H%M%S")), mode='I')


class GANMonitor(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        generated_images = self.model.generator(noise)
        generated_images *= (generated_images * 127.5) + 127.5
        output = generated_images[0].cpu().detach().numpy()
        img = keras.utils.array_to_img(output)
        img.save("./generated_images/generated_image_{}.png".format(epoch))

gan.fit(dataset, epochs=20, callbacks=[GANMonitor()])
writer.close()