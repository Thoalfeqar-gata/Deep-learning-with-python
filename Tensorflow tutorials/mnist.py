from keras.datasets import mnist
from keras.layers import Dropout, Dense, Flatten, Softmax
from keras.models import Sequential
from keras import optimizers, losses

(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
train_data, test_data = train_data/255.0, test_data/255.0

logits_model = Sequential([
    Flatten(input_shape = (28, 28)),
    Dense(128, 'relu'),
    Dropout(0.2),
    Dense(10, 'linear')
])

logits_model.compile(optimizers.Adam(), losses.SparseCategoricalCrossentropy(from_logits = True), metrics = ['accuracy'])
logits_model.fit(train_data, train_labels, epochs = 10)
logits_model.evaluate(test_data, test_labels)

probability_model = Sequential([
    logits_model,
    Softmax()
])

print(probability_model.predict(test_data[:5]))