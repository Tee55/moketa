import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras_core import layers, Model
import keras_core as keras

image_size = 64
latent_size = 128
batch_size = 128
epochs = 25
lr = 0.0002

def build_discriminator():
    model = keras.Sequential([
        layers.Input(shape=(image_size, image_size, 3)),
        layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        
        layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        
        layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        
        layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        
        layers.Conv2D(1, (4, 4), padding='valid', use_bias=False),
        layers.Flatten(),
        layers.Activation('sigmoid')
    ])
    return model

def build_generator():
    model = keras.Sequential([
        layers.Input(shape=(latent_size,)),
        layers.Reshape((1, 1, latent_size)),
        
        layers.Conv2DTranspose(512, (4, 4), strides=(1, 1), padding='valid', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        layers.Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', use_bias=False),
        layers.Activation('tanh')
    ])
    return model

discriminator = build_discriminator()
discriminator.summary()

generator = build_generator()
generator.summary()

cross_entropy = keras.losses.BinaryCrossentropy()

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = keras.optimizers.Adam(lr, beta_1=0.5)
discriminator_optimizer = keras.optimizers.Adam(lr, beta_1=0.5)

class GAN(keras.Model):
    def __init__(self):
        super().__init__()
        self.d_loss_tracker = keras.metrics.Mean(name="d_loss")
        self.g_loss_tracker = keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_tracker, self.g_loss_tracker]

    @tf.function
    def train_step(self, images):
        noise = tf.random.normal([batch_size, latent_size])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)
            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        # Update metrics and return their value.
        self.d_loss_tracker.update_state(disc_loss)
        self.g_loss_tracker.update_state(gen_loss)
        return {
            "d_loss": self.d_loss_tracker.result(),
            "g_loss": self.g_loss_tracker.result(),
        }

class GANMonitor(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        sample_noise = tf.random.normal([1, latent_size])
        generated_images = generator(sample_noise)
        generated_images = 0.5 * generated_images + 0.5 
        output = generated_images[0].numpy()
        img = keras.utils.array_to_img(output)
        img.save("./generated_images/generated_image_{}.png".format(epoch))

if __name__ == "__main__":
    data_dir = 'data/train/faces'
    train_dataset = keras.utils.image_dataset_from_directory(
        data_dir, 
        image_size=(image_size, image_size), 
        batch_size=batch_size, 
        label_mode=None
    )
    train_dataset = train_dataset.map(lambda x: (x - 127.5) / 127.5)

    gan = GAN()
    gan.compile()
    gan.fit(train_dataset, epochs=epochs, callbacks=[GANMonitor()])

    num_samples = 9
    sample_noise = tf.random.normal([num_samples, latent_size])
    sample_images = generator(sample_noise)
    sample_images = 0.5 * sample_images + 0.5  # Denormalize
    sample_images = np.clip(sample_images, 0, 1)  # Clip values to [0, 1]

    fig, axs = plt.subplots(3, 3, figsize=(8, 8))
    for i in range(num_samples):
        axs[i // 3, i % 3].imshow(sample_images[i])
        axs[i // 3, i % 3].axis('off')
    plt.show()