from keras.applications.vgg19 import VGG19, preprocess_input
from keras import backend as K
from scipy.optimize import fmin_l_bfgs_b
import keras, cv2, numpy as np, tensorflow as tf, time
tf.compat.v1.disable_eager_execution()


target_image_path = 'Chapter 8/Neural style transfer/images/house.jpg'
style_reference_image_path = 'Chapter 8/Neural style transfer/images/starry_night.png'

height, width = cv2.imread(target_image_path).shape[0:2]
img_height = 400
img_width = int(width * img_height / height)

def preprocess_image(image_path):
    img = cv2.resize(cv2.imread(image_path), (img_width, img_height))
    img = np.expand_dims(img, axis = 0)
    img = preprocess_input(img)
    return img

def deprocess_image(img):
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def content_loss(base_activation, combination_activation):
    return K.sum(K.square(combination_activation - base_activation))

def gram_matrix(feature_map):
    features = K.batch_flatten(K.permute_dimensions(feature_map, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

def style_loss(style_activation, combination_activation):
    S = gram_matrix(style_activation)
    C = gram_matrix(combination_activation)
    channels = 3
    size = img_height * img_width
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

def total_variation_loss(feature_map):
    a = K.square(
        feature_map[:, :img_height - 1, :img_width - 1, :] - 
        feature_map[:, 1:, :img_width - 1, :]
    )
    
    b = K.square(
        feature_map[:, :img_height - 1, :img_width - 1, :] -
        feature_map[:, :img_height - 1, 1:, :]
    )
    return K.sum(K.pow(a + b, 1.25))


target_image = K.constant(preprocess_image(target_image_path))
style_reference_image = K.constant(preprocess_image(style_reference_image_path))
combination_image = K.placeholder((1, img_height, img_width, 3))

input_tensor = K.concatenate([target_image, style_reference_image, combination_image], axis = 0)

model = VGG19(input_tensor = input_tensor, include_top = False)

output_dict = dict([(layer.name, layer.output) for layer in model.layers])
content_layer = 'block5_conv2'
style_layers = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
    'block5_conv1',
]
total_variation_weight = 1e-4
style_weight = 1.
content_weight = 0.025

loss = K.variable(0.)
layer_features = output_dict[content_layer]
target_image_features = layer_features[0, :, :, :]
combination_image_features = layer_features[2, :, :, :]
loss = loss + content_weight * content_loss(target_image_features, combination_image_features)

for layer_name in style_layers:
    layer_features = output_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_image_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_features, combination_image_features)
    loss += (style_weight / len(style_layers)) * sl

loss += total_variation_weight * total_variation_loss(combination_image)

grads = K.gradients(loss, combination_image)[0]

fetch_loss_and_grads = K.function([combination_image], [loss, grads])

class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_value = None
    
    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, img_height, img_width, 3))
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype(np.float64)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value
    
    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values
    
evaluator = Evaluator()

iterations = 50
x = preprocess_image(target_image_path)
x = x.flatten()

for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.perf_counter()
    
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x, evaluator.grads, maxfun = 20)
    
    print('Current loss value:', min_val)
    img = x.copy().reshape((img_height, img_width, 3))
    img = deprocess_image(img)
    filename = f'Chapter 8/Neural style transfer/outputs/Result at iteration {i}.png'
    cv2.imwrite(filename, img)
    end_time = time.perf_counter()
    print(f'Iteration {i} took {end_time - start_time} to finish')
