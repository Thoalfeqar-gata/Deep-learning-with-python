import keras, tensorflow as tf, numpy as np, cv2
from keras.applications import VGG16
from keras import backend as K
tf.compat.v1.disable_eager_execution()

model = VGG16(include_top = False)
model.summary()

def deprocess_image(img):
    img -= img.mean()
    img /= (img.std() + 1e-5)
    img *= 0.1
    
    img += 0.5
    img = np.clip(img, 0, 1)
    
    img *= 255
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def generate_pattern(layer_name, filter_index, size = 150):
    layer_output = model.get_layer(layer_name).output

    loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    iterate = K.function([model.input], [loss, grads])

    loss_value, grads_value = iterate([np.zeros((1, size, size, 3))])

    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
    
    return deprocess_image(input_img_data[0])

layer_name = 'block4_conv1'
size = 128
margin = 5
axis_image_count = 8
results = np.zeros((axis_image_count * size + (axis_image_count - 1) * margin, axis_image_count * size + (axis_image_count - 1) * margin, 3), dtype = np.uint8)

for i in range(axis_image_count):
    for j in range(axis_image_count):
        filter_image = generate_pattern(layer_name, i + (j * axis_image_count), size = size)
        
        horizontal_start = i * size + i * margin
        horizontal_end = horizontal_start + size
        
        vertical_start = j * size + j * margin
        vertical_end = vertical_start + size
        results[horizontal_start : horizontal_end, vertical_start : vertical_end] = filter_image

cv2.imshow('Filter patterns', results)
cv2.waitKey(0)