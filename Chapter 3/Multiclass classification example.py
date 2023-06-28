import numpy as np
from keras.datasets import reuters
from keras import models, layers
from matplotlib import pyplot as plt

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words = 10000)

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

model = models.Sequential([
    layers.Dense(64, 'relu'),
    layers.Dense(64, 'relu'),
    layers.Dense(46, 'softmax')
])

model.compile('rmsprop', 'sparse_categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(x_train, train_labels, epochs = 20, batch_size = 512, validation_split = 0.2)
history_dict = history.history

training_loss = history_dict['loss']
training_accuracy = history_dict['accuracy']
val_loss = history_dict['val_loss']
val_accuracy = history_dict['val_accuracy']
epochs = range(1, len(training_loss) + 1)

plt.figure()
plt.title('Training loss and accuracy')
plt.plot(epochs, training_loss, 'bo', label = 'Training loss')
plt.plot(epochs, training_accuracy, 'b', label = 'Training accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.figure()
plt.title('Validation loss and accuracy')
plt.plot(epochs, val_loss, 'bo', label = 'Validation loss')
plt.plot(epochs, val_accuracy, 'b', label = 'Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()