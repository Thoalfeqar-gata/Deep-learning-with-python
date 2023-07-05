from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
import numpy as np, cv2, tensorflow as tf
from keras import backend as K
tf.compat.v1.disable_eager_execution()

model = VGG16()
model.summary()
image_path = 'Chapter 5/visualization/images/elephant2.jpg'
image = cv2.resize(cv2.imread(image_path), (224, 224))
image = image.astype(np.float32)
image = np.expand_dims(image, 0)
image = preprocess_input(image)
index = np.argmax(model.predict(image)[0])


#grad_CAM
elephant_output = model.output[:, index]
last_conv_layer = model.get_layer('block5_conv3')
grads = K.gradients(elephant_output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis = (0, 1, 2))
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([image])

for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis = -1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

image = cv2.imread(image_path)
heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + image
superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
cv2.imshow('super imposed image', superimposed_img)
cv2.waitKey(0)
