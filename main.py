import os
os.environ["KERAS_BACKEND"] = "torch"
import keras_core as keras
import matplotlib.pyplot as plt
import time
import imageio
import keras_core as keras
import torch
print(torch.cuda.get_device_name(0))

class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_loss_tracker = keras.metrics.Mean(name="d_loss")
        self.g_loss_tracker = keras.metrics.Mean(name="g_loss")
        self.seed_generator = keras.random.SeedGenerator(1337)
        self.built = True

    @property
    def metrics(self):
        return [self.d_loss_tracker, self.g_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(real_images, tuple):
            real_images = real_images[0]
        # Sample random points in the latent space
        batch_size = real_images.shape[0]
        random_latent_vectors = keras.random.normal(
            shape=(batch_size, self.latent_dim), seed=self.seed_generator
        )

        # Decode them to fake images
        generated_images = self.generator(random_latent_vectors)

        # Combine them with real images
        real_images = torch.tensor(real_images, device=device)
        combined_images = torch.concat([generated_images, real_images], axis=0)

        # Assemble labels discriminating real from fake images
        labels = torch.concat(
            [
                torch.ones((batch_size, 1), device=device),
                torch.zeros((batch_size, 1), device=device),
            ],
            axis=0,
        )
        # Add random noise to the labels - important trick!
        labels += 0.05 * keras.random.uniform(labels.shape, seed=self.seed_generator)

        # Train the discriminator
        self.zero_grad()
        predictions = self.discriminator(combined_images)
        d_loss = self.loss_fn(labels, predictions)
        d_loss.backward()
        grads = [v.value.grad for v in self.discriminator.trainable_weights]
        with torch.no_grad():
            self.d_optimizer.apply(grads, self.discriminator.trainable_weights)

        # Sample random points in the latent space
        random_latent_vectors = keras.random.normal(
            shape=(batch_size, self.latent_dim), seed=self.seed_generator
        )

        # Assemble labels that say "all real images"
        misleading_labels = torch.zeros((batch_size, 1), device=device)

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        self.zero_grad()
        predictions = self.discriminator(self.generator(random_latent_vectors))
        g_loss = self.loss_fn(misleading_labels, predictions)
        grads = g_loss.backward()
        grads = [v.value.grad for v in self.generator.trainable_weights]
        with torch.no_grad():
            self.g_optimizer.apply(grads, self.generator.trainable_weights)

        # Update metrics and return their value.
        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)
        return {
            "d_loss": self.d_loss_tracker.result(),
            "g_loss": self.g_loss_tracker.result(),
        }

latent_dim = 300
def create_generator():
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

#plot_dataset(16)

generator = create_generator()
discriminator = create_discriminator()
noise = keras.random.normal(shape=(1, latent_dim))

writer = imageio.get_writer(
    './generated_images/{}.gif'.format(time.strftime("%Y%M%d-%H%M%S")), mode='I')


class GANMonitor(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        generated_images = self.model.generator(noise, training=False)
        generated_images = (generated_images * 255) + 255
        output = generated_images[0].detach().cpu().numpy().astype("uint8")
        img = keras.utils.array_to_img(output)
        img.save("./generated_images/generated_image_{}.png".format(epoch))
        writer.append_data(img)

gan = GAN(discriminator=discriminator,
          generator=generator, latent_dim=latent_dim)
gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
)
gan.fit(dataset_preprocess, epochs=50, callbacks=[GANMonitor()])
writer.close()