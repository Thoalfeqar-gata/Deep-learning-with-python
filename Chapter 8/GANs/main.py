from keras import layers
from keras.datasets import cifar10
from keras.optimizers import RMSprop
from keras.models import Model
from keras.preprocessing import image
import numpy as np, keras, os, cv2

latent_dim = 32
height = 32
width = 32
channels = 3

generator_input = layers.Input(shape = (latent_dim,))
x = layers.Dense(128 * 16 * 16)(generator_input)
x = layers.LeakyReLU()(x)
x = layers.Reshape((16, 16, 128))(x)

x = layers.Conv2D(256, 5, padding = 'same')(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2DTranspose(256, 4, strides = 2, padding = 'same')(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(256, 5, padding = 'same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(256, 5, padding = 'same')(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(channels, 7, activation = 'tanh', padding = 'same')(x)
generator = Model(generator_input, x)
generator.summary()

discriminator_input = layers.Input(shape = (height, width, channels))
x = layers.Conv2D(128, 3)(discriminator_input)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides = 2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides = 2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides = 2)(x)
x = layers.LeakyReLU()(x)
x = layers.Flatten()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(1, activation = 'sigmoid')(x)

discriminator = Model(discriminator_input, x)
discriminator.summary()

discriminator_optimizer = RMSprop(
    learning_rate = 0.0008,
    clipvalue = 1.0,
    decay = 1e-8
)

discriminator.compile(optimizer = discriminator_optimizer, loss = 'binary_crossentropy')

discriminator.trainable = False
gan_input = keras.Input(shape = (latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = Model(gan_input, gan_output)
gan_optimizer = RMSprop(learning_rate = 0.0004, clipvalue = 1.0, decay = 1e-8)
gan.compile(optimizer = gan_optimizer, loss = 'binary_crossentropy')

(x_train, y_train), (_, _) = cifar10.load_data()
x_train = x_train[y_train.flatten() == 6]
x_train = x_train.astype(np.float32) / 255.0
x_train = x_train[:, :, :, ::-1]

iterations = 50000
batch_size = 20
save_dir = 'Chatper 8/GANs/images/'

start = 0
for step in range(iterations):
    random_latent_vectors = np.random.normal(size = (batch_size, latent_dim))
    generated_images = generator.predict(random_latent_vectors)
    
    stop = start + batch_size
    real_images = x_train[start:stop]
    combined_images = np.concatenate([generated_images, real_images])
    labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
    
    labels += 0.05 * np.random.random(labels.shape)
    
    d_loss = discriminator.train_on_batch(combined_images, labels)
    
    random_latent_vectors = np.random.normal(size = (batch_size, latent_dim))
    
    misleading_targets = np.zeros((batch_size, 1))
    a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)
    
    start += batch_size
    if start > len(x_train) - batch_size:
        start = 0
    
    if step % 100 == 0:
        gan.save_weights('Chapter 8/GANs/models/gan.h5')

        print('Discriminator loss:', d_loss)
        print('adversarial loss:', a_loss)
        
        img = np.clip(generated_images[0] * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(f'Chapter 8/GANs/images/generated {step}.png', img)
        
        real_img = np.clip(real_images[0] * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(f'Chapter 8/GANs/images/real {step}.png', real_img)
        
        