import os
os.environ["KERAS_BACKEND"] = "torch"
import keras_core as keras
import matplotlib.pyplot as plt
import time
from GAN import GAN
import imageio
import keras_core as keras
import tensorflow as tf


dataset = keras.utils.image_dataset_from_directory(
    "face_images", image_size=(64, 64), batch_size=32
)

def process(image, label):
    image = tf.cast(image/255. ,tf.float32)
    return image, label

dataset = dataset.map(process)

def plot_dataset(num_samples):
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        for i in range(num_samples):
            plt.subplot(4, 4, i+1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.axis("off")
    plt.show()

#plot_dataset(16)

def create_discriminator():
    model = keras.Sequential(name="Discriminator")
    model.add(keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, padding="same", activation='relu', input_shape=(64, 64, 3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(filters=128, kernel_size=4, strides=2, padding="same", activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(filters=256, kernel_size=4, strides=2, padding="same", activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=1, activation="sigmoid", name="discriminator_dense"))
    print(model.summary())
    return model

latent_dim = 128
def create_generator():
    model = keras.Sequential(name="Generator")
    model.add(keras.layers.Dense(4 * 4 * 256, input_shape=(latent_dim, )))
    model.add(keras.layers.Reshape((4, 4, 256)))
    model.add(keras.layers.Conv2DTranspose(filters=128, kernel_size=4, strides=2, padding="same", activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2DTranspose(filters=128, kernel_size=4, strides=2, padding="same", activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2DTranspose(filters=128, kernel_size=4, strides=2, padding="same", activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2DTranspose(filters=3, kernel_size=4, strides=2, padding="same", activation='relu'))
    print(model.summary())
    return model


generator = create_generator()
discriminator = create_discriminator()

def plot_generator():
    noise = keras.random.normal(shape=(1, latent_dim))
    generated_image = generator(noise, training=False)
    plt.imshow(generated_image[0, :, :, 0])
    plt.show()

#plot_generator()

gan = GAN(discriminator=discriminator,
          generator=generator, latent_dim=latent_dim)
gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
)

writer = imageio.get_writer(
    './generated_images/{}.gif'.format(time.strftime("%Y%M%d-%H%M%S")), mode='I')


class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=1, latent_dim=latent_dim):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = keras.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()
        image = keras.utils.array_to_img(generated_images[0])
        image.save("./generated_images/generated_image_%03d.png" % (epoch))
        writer.append_data(image)


gan.fit(dataset, epochs=1, callbacks=[GANMonitor()])
writer.close()