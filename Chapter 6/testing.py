from keras.datasets import imdb
from keras.utils import pad_sequences
from keras.layers import Dense, SimpleRNN, Embedding, LSTM, Dropout, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from matplotlib import pyplot as plt
import keras, numpy as np

max_words = 10000
max_review_length = 1000
batch_size = 128
embedding_size = 128

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = max_words)

x_train = pad_sequences(x_train, maxlen = max_review_length, padding = 'post', truncating = 'post')
x_test = pad_sequences(x_test, maxlen = max_review_length, padding = 'post', truncating = 'post')
print(x_train.shape, x_test.shape)

model = keras.models.Sequential([
    Embedding(input_dim = max_words, output_dim = embedding_size, input_length = max_review_length),
    Conv1D(32, 7, activation = 'relu'),
    MaxPooling1D(5),
    Conv1D(32, 7, activation = 'relu'),
    GlobalMaxPooling1D(),
    Dense(1, 'sigmoid')
])

model.summary()
model.compile('rmsprop', 'binary_crossentropy', metrics = ['accuracy'])
history = model.fit(x_train, y_train, batch_size = batch_size, epochs = 15, validation_split = 0.2)
model.evaluate(x_test, y_test)

history_dict = history.history
loss = history_dict.get('loss')
val_loss = history_dict.get('val_loss')
accuracy = history_dict.get('accuracy')
val_accuracy = history_dict.get('val_accuracy')
x_axis = range(1, len(loss) + 1)

plt.figure('Model loss')
plt.plot(x_axis, loss, 'bo', label = 'loss')
plt.plot(x_axis, val_loss, 'b', label = 'validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc = 'best')

plt.figure('Model accuracy')
plt.plot(x_axis, accuracy, 'bo', label = 'accuracy')
plt.plot(x_axis, val_accuracy, 'b', label = 'validation accuracy')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc = 'best')
plt.show()


