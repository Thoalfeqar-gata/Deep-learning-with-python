from keras.datasets import mnist
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential

(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
train_data, test_data = train_data/255.0, test_data/255.0

model = Sequential([
    Conv2D(32, (3, 3), activation = 'relu', input_shape = (28, 28, 1), kernel_initializer = 'ones'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation = 'relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation = 'relu'),
    Flatten(),
    Dense(64, 'relu'),
    Dense(10, 'softmax')
])

model.compile('rmsprop', 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(train_data, train_labels, batch_size = 64, epochs = 5)
model.evaluate(test_data, test_labels)