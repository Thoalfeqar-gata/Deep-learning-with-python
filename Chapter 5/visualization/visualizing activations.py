import keras, cv2, numpy as np
from matplotlib import pyplot as plt

model = keras.models.load_model('Chapter 5/cats and dogs classification/models/cats_and_dogs_model.h5')
#load, crop, resize, and normalize the image
image = cv2.resize(cv2.imread('Chapter 5/cats and dogs classification/data/test/7.jpg'), (200, 200))
image = image / 255.0
image = np.expand_dims(image, axis = 0)

layers_outputs = [layer.output for layer in model.layers[:10]]
layers_names = [layer.name for layer in model.layers[:10]]
activations_model = keras.models.Model(inputs = model.input, outputs = layers_outputs)

activations = activations_model.predict(image)
for activation, layer_name in zip(activations, layers_names):
    activation = activation[0]
    filter_outputs = [activation[:, :, i] for i in range(activation.shape[2])]
    for filter_output in filter_outputs:
        filter_output = cv2.resize(filter_output, (500, 500), interpolation = cv2.INTER_LINEAR)
        filter_output *= 255  
        filter_output = np.clip(filter_output * 4, 0, 255) #multiply by 255 to increase the brightness and litmit the max value to 255
        cv2.imshow(layer_name, filter_output.astype(np.uint8))
        cv2.waitKey(0)
        
    cv2.destroyWindow(layer_name)
        