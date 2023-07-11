from keras.models import Model
from keras import layers
import numpy as np

x = layers.Input(shape = (224, 224, 3))
y = layers.Conv2D(128, 3, activation = 'relu', padding = 'same')(x)
y = layers.Conv2D(128, 3, activation = 'relu', padding = 'same')(y)
y = layers.MaxPooling2D(2, strides = 2)(y)

residual = layers.Conv2D(128, 1, strides = 2)(x)

y = layers.add([y, residual])

model = Model(inputs = [x], outputs = [y])
model.summary()