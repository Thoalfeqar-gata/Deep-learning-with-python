from keras.datasets import fashion_mnist
from keras.layers import Dropout, Dense, Flatten
from keras.models import Sequential
from keras import optimizers, losses
import cv2, numpy as np

(train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()
train_data, test_data = train_data/255.0, test_data/255.0
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model = Sequential([
    Flatten(input_shape = (28, 28)),
    Dense(128, 'relu'),
    Dropout(0.2),
    Dense(10, 'softmax')
])

model.compile(optimizers.Adam(), losses.SparseCategoricalCrossentropy(), metrics = ['accuracy'])
model.fit(train_data, train_labels, epochs = 20)
model.evaluate(test_data, test_labels)

for image in test_data:
    prediction = model.predict(np.expand_dims(image, axis = 0), verbose = 0)[0]
    prediction = np.argmax(prediction, -1)
    name = class_names[prediction]
    
    image = image*255
    image = image.astype(np.uint8)
    image = cv2.resize(image, (0, 0), fx = 8, fy = 8)
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyWindow(name)