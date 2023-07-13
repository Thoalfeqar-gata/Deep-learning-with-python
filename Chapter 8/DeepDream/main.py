from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras import backend as K
import tensorflow as tf
import numpy as np, scipy, cv2
tf.compat.v1.disable_eager_execution()
K.set_learning_phase(0)

model = InceptionV3(include_top = False)
layer_contributions = {
    'mixed2' : 0.2,
    'mixed3' : 3.,
    'mixed4' : 2.,
    'mixed5' : 1.5
}

layer_dict = dict([(layer.name, layer) for layer in model.layers])

loss = K.variable(0.)
for layer_name in layer_contributions:
    coeff = layer_contributions[layer_name]
    activation = layer_dict[layer_name].output
    
    scaling = K.prod(K.cast(K.shape(activation), 'float32')) # finds how many elements are in the activation by mutltiplying the activation shape
    loss = loss + coeff * K.sum(K.square(activation[:, 2:-2, 2:-2, :])) / scaling

dream = model.input
grads = K.gradients(loss, dream)[0]
grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)

outputs = [loss, grads]
fetch_loss_and_grads = K.function([dream], outputs)

def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_values = outs[1]
    return loss_value, grad_values


def gradient_ascent(x, iterations, step, max_loss = None):
    
    for i in range(iterations):
        loss_value, grad_values = eval_loss_and_grads(x)
        if max_loss is not None and loss_value > max_loss:
            break
        print('...Loss value at', i, ':', loss_value)
        x += step * grad_values
    return x


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (0, 0), fx = 1/2, fy = 1/2)
    cv2.imwrite('Chapter 8/DeepDream/output_images/original.jpg', img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis = 0)
    img = preprocess_input(img)
    return img

def deprocess_image(image):
    if K.image_data_format() == 'channels-first':
        image = image.reshape(3, image.shape[2], image.shape[3])
        image = np.transpose(image, (1, 2, 0))    
    else:
        image = image.reshape((image.shape[1], image.shape[2], 3))
    
    image /= np.max(image)
    image += 0.15
    image *= 255.
    image = np.clip(image, 0, 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def resize_img(img, size):
    img = np.copy(img)
    factors = (1,
    float(size[0]) / img.shape[1],
    float(size[1]) / img.shape[2],
    1)
    return scipy.ndimage.zoom(img, factors, order=1)


step = 0.01
num_octaves = 3
octave_scale = 1.4
iterations = 20
max_loss = None

img = preprocess_image('Chapter 8/DeepDream/images/dog.jpeg')
original_shape = img.shape[1:3]
successive_shapes = [original_shape]
for i in range(1, num_octaves):
    shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
    successive_shapes.append(shape)
successive_shapes = successive_shapes[::-1]

original_img = np.copy(img)
shrunk_original_img = resize_img(img, successive_shapes[0])

for shape in successive_shapes:
    print('Processing image shape', shape)
    img = resize_img(img, shape)
    img = gradient_ascent(x = img, iterations = iterations, step = step, max_loss = max_loss)
    
    upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
    same_size_original = resize_img(original_img, shape)
    lost_detail = same_size_original - upscaled_shrunk_original_img
    
    img += lost_detail
    shrunk_original_img = resize_img(original_img, shape)
    output_image = deprocess_image(img)
    print(output_image.shape)
    cv2.imwrite(f'Chapter 8/DeepDream/output_images/image {shape}.jpg', output_image)

