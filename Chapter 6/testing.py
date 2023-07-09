import keras
from keras import utils
from keras.datasets import imdb
from keras.layers import Embedding, Flatten, Dense

max_features = 10000
maxlen = 100

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = max_features)

x_train = utils.pad_sequences(x_train, maxlen=maxlen, padding = 'post', truncating = 'post')
x_test = utils.pad_sequences(x_test, maxlen=maxlen, padding = 'post', truncating = 'post')

model = keras.models.Sequential([
    Embedding(max_features, 16, input_length = maxlen),
    Flatten(),
    Dense(1, activation = 'sigmoid')
])

model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()

model.fit(x_train, y_train, epochs = 10, batch_size = 32, validation_split = 0.2)