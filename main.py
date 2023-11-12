import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras_core as keras
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np

img_width = 64
img_height = 64
channels = 3
img_shape = (img_width, img_height, channels)
latent_dim = 200
output_path = 'generated_images/'
dir_path = 'data/train/faces'

def build_generator():

    generator = keras.Sequential(name="Generator")

    generator.add(keras.layers.Input(shape=(latent_dim, )))

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

    generator.add(keras.layers.Conv2D(filters=channels, kernel_size=4, padding="same", activation='tanh'))
    print(generator.summary())

    return generator

def build_discriminator():

    discriminator = keras.Sequential(name="Discriminator")

    discriminator.add(keras.layers.Input(shape=(img_width, img_height, channels)))

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

class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_loss_tracker = keras.metrics.Mean(name="d_loss")
        self.g_loss_tracker = keras.metrics.Mean(name="g_loss")
        self.seed_generator = keras.random.SeedGenerator(1337)

    @property
    def metrics(self):
        return [self.d_loss_tracker, self.g_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]
        # Sample random points in the latent space
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = keras.random.normal(
            shape=(batch_size, self.latent_dim), seed=self.seed_generator
        )

        # Decode them to fake images
        generated_images = self.generator(random_latent_vectors)

        # Combine them with real images
        combined_images = tf.concat([generated_images, real_images], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        # Add random noise to the labels - important trick!
        labels += 0.05 * keras.random.uniform(
            tf.shape(labels), seed=self.seed_generator
        )

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply(grads, self.discriminator.trainable_weights)

        # Sample random points in the latent space
        random_latent_vectors = keras.random.normal(
            shape=(batch_size, self.latent_dim), seed=self.seed_generator
        )

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply(grads, self.generator.trainable_weights)

        # Update metrics and return their value.
        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)
        return {
            "d_loss": self.d_loss_tracker.result(),
            "g_loss": self.g_loss_tracker.result(),
        }

class GANMonitor(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        r, c = 5, 5
        noise = keras.random.normal(shape=(r * c, latent_dim))
        generated_images = self.model.generator(noise)
        generated_images = 0.5 * generated_images + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                img = keras.utils.array_to_img(generated_images[cnt])
                axs[i, j].imshow(img)
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(os.path.join(output_path, "output_{}.png".format(epoch)))
        plt.close()


if __name__ == '__main__':

    for filename in os.listdir(output_path):
        if filename.endswith('.png'):
            os.remove(os.path.join(output_path, filename))

    """ X_train = []
    for filename in os.listdir(dir_path)[:500]:
        image = keras.utils.load_img(os.path.join(dir_path, filename), target_size=(img_height, img_width))
        image = keras.utils.img_to_array(image)
        X_train.append(image)

    X_train = np.array(X_train)
    X_train = X_train / 127.5 - 1. """

    X_train = keras.utils.image_dataset_from_directory(
                dir_path,
                labels=None,
                color_mode="rgb",
                batch_size=50,
                image_size=(img_width, img_height),
                shuffle=True)

    normalization_layer = keras.layers.Rescaling(1./127.5, offset=-1)
    X_train = X_train.map(lambda x: normalization_layer(x))

    discriminator = build_discriminator()
    generator = build_generator()
    gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
    gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
        loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
    )

    gan.fit(X_train, epochs=50, callbacks=[GANMonitor()])
