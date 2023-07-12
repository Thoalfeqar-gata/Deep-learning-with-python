from keras.models import Model
from keras import layers
from keras.datasets import imdb
from keras.utils import pad_sequences, plot_model
import keras
import numpy as np

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = 2000)
x_train = pad_sequences(x_train, maxlen = 500)
x_test = pad_sequences(x_test, maxlen = 500)

model = keras.models.Sequential([
    layers.Embedding(2000, 128, input_length = 500),
    layers.Conv1D(32, 7, activation = 'relu', name = 'first_conv'),
    layers.BatchNormalization(name = 'batch_normalization'),
    layers.MaxPooling1D(5),
    layers.Conv1D(32, 7, activation = 'relu'),
    layers.GlobalMaxPooling1D(),
    layers.Dense(1, activation = 'sigmoid')
])

model.summary()
model.compile('rmsprop', 'binary_crossentropy', metrics = ['accuracy'])

callbacks = [
    keras.callbacks.TensorBoard(
        log_dir = 'Chapter 7/log',
        histogram_freq = 1,
        embeddings_freq = 1,
    )
]

history = model.fit(x_train, y_train, epochs = 5, batch_size = 128, validation_split = 0.2, callbacks = callbacks)

