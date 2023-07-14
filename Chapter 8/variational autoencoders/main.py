from keras import layers
from keras import backend as K
from keras.models import Model
from keras.utils import plot_model
from keras.datasets import mnist
from scipy.stats import norm
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import keras
import numpy as np, tensorflow as tf, os, cv2
tf.compat.v1.disable_eager_execution()

size = 160
img_shape = (size, size, 3)
batch_size = 16
latent_dim = 10

input_img = layers.Input(shape = img_shape)
x = layers.Conv2D(32, 3, padding = 'same', activation = 'relu')(input_img)
x = layers.Conv2D(64, 3, padding = 'same', activation = 'relu', strides = (2, 2))(x)
x = layers.Conv2D(64, 3, padding = 'same', activation = 'relu')(x)
x = layers.Conv2D(64, 3, padding = 'same', activation = 'relu')(x)
shape_before_flattening = K.int_shape(x)

x = layers.Flatten()(x)
x = layers.Dense(32, activation = 'relu')(x)

z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape = (K.shape(z_mean)[0], latent_dim), mean = 0., stddev = 1.)
    return z_mean + K.exp(z_log_var) * epsilon

z = layers.Lambda(sampling)([z_mean, z_log_var])

decoder_input = layers.Input(K.int_shape(z)[1:])

x = layers.Dense(np.prod(shape_before_flattening[1:]), 'relu')(decoder_input)
x = layers.Reshape(shape_before_flattening[1:])(x)
x = layers.Conv2DTranspose(32, 3, padding = 'same', activation = 'relu', strides = (2, 2))(x)
x = layers.Conv2D(3, 3, padding = 'same', activation = 'sigmoid')(x)

decoder = Model(decoder_input, x)
z_decoded = decoder(z)

class CustomVariationalLayer(keras.layers.Layer):
    def vae_loss(self, x, z_decoded): #x is the input image
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
        kl_loss = -5e-4 * K.mean(
            1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis = -1
        )
        return K.mean(xent_loss + kl_loss)
    
    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs = inputs)
        return x

y = CustomVariationalLayer()([input_img, z_decoded])

vae = Model(input_img, y)
vae.compile(optimizer = 'rmsprop', loss = None)
vae.summary()
plot_model(vae, to_file = 'Chapter 8/variational autoencoders/model/model.png', expand_nested = True, dpi = 128)

images = []
for dir, dirnames, filenames in os.walk('Chapter 8/variational autoencoders/Data/lfw_funneled'): #the data is not included because of its large size
    if len(filenames) <= 0:
        continue
    
    for filename in filenames:
        image = cv2.resize(cv2.imread(os.path.join(dir, filename)), (size, size))
        images.append(image)
    
total_images = len(images)
test_size = 1000
train_size = total_images - test_size
x_train = np.asarray(images[:train_size]) / 255.
x_test = np.asarray(images[train_size:]) / 255.

vae.fit(x = x_train, y = None, shuffle = True, epochs = 10, batch_size = batch_size, validation_data = (x_test, None))

n = 7
image_size = size
figure = np.zeros((image_size * n, image_size * n, 3))
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_y):
    for j, xi in enumerate(grid_x):
        z_sample = np.array([[xi, yi, xi, yi, xi, yi, xi, yi, xi, yi]])
        x_decoded = decoder.predict(z_sample)
        image = x_decoded[0].reshape(image_size, image_size, 3)
        figure[i * image_size : (i + 1) * image_size, j * image_size : (j + 1) * image_size, :] = image
        
cv2.imshow('hi', figure)
cv2.waitKey(0)
plt.figure(figsize = (15, 15))
plt.imshow(figure)
plt.show()