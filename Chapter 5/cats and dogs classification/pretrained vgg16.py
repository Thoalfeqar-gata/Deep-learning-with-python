import os, shutil, numpy as np, random, keras, cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.applications import VGG16
from keras.optimizers import RMSprop
from matplotlib import pyplot as plt

data_path = 'Chapter 5/cats and dogs classification/data'
train_dir = os.path.join(data_path, 'train')
test_dir = os.path.join(data_path, 'test')
validation_dir = os.path.join(data_path, 'validation')
train_size = 20000
test_size = 12500

if not os.path.exists(os.path.join(train_dir, 'cats')):
    os.mkdir(os.path.join(train_dir, 'cats'))
    
if not os.path.exists(os.path.join(train_dir, 'dogs')):
    os.mkdir(os.path.join(train_dir, 'dogs'))

validation_size = 5000

if not os.path.exists(validation_dir):
    os.mkdir(os.path.join(data_path, 'validation'))
    
    train_files = os.listdir(train_dir)
    random.shuffle(train_files)

    for i, file in enumerate(train_files):
        shutil.move(os.path.join(train_dir, file), os.path.join(validation_dir, file))
        if i >= validation_size - 1:
            break
    os.mkdir(os.path.join(validation_dir, 'cats'))
    os.mkdir(os.path.join(validation_dir, 'dogs'))
    
    for filename in os.listdir(validation_dir):
        if filename.endswith('.jpg'):
            if 'dog' in filename:
                shutil.move(os.path.join(validation_dir, filename), os.path.join(validation_dir, 'dogs', filename))
            elif 'cat' in filename:
                shutil.move(os.path.join(validation_dir, filename), os.path.join(validation_dir, 'cats', filename))

for filename in os.listdir(train_dir):
    if filename.endswith('.jpg'):
        if 'dog' in filename:
            shutil.move(os.path.join(train_dir, filename), os.path.join(train_dir, 'dogs', filename))
        elif 'cat' in filename:
            shutil.move(os.path.join(train_dir, filename), os.path.join(train_dir, 'cats', filename))


train_datagen = ImageDataGenerator(
    rescale = 1/255.0,                            
    # rotation_range=40,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # shear_range=0.2,
    # zoom_range=0.2,
    # horizontal_flip=True    
)
batch_size = 32
train_generator = train_datagen.flow_from_directory(
    directory = train_dir, 
    target_size = (200, 200),
    batch_size = batch_size,
    class_mode = 'binary'
)

validation_datagen = ImageDataGenerator(rescale = 1/255.0)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size = (200, 200),
    batch_size = batch_size,
    class_mode = 'binary'
)

base_convnet = VGG16(include_top = False, input_shape = (200, 200, 3))

model = Sequential([
    base_convnet,
    Flatten(),
    Dense(256, activation = 'relu'),
    Dense(1, 'sigmoid')
])

for layer in base_convnet.layers:
    layer.trainable = False
    
model.compile(RMSprop(), 'binary_crossentropy', metrics = ['accuracy'])
history = model.fit(train_generator, steps_per_epoch = train_size / batch_size, epochs = 25, validation_data = validation_generator, validation_steps = validation_size / batch_size)
model.save('Chapter 5/cats and dogs classification/models/cats_and_dogs_model_vgg16.h5')

history_dict = history.history
training_loss = history_dict['loss']
validation_loss = history_dict['val_loss']
training_accuracy = history_dict['accuracy']
validation_accuracy = history_dict['val_accuracy']
x_axis = range(1, len(training_loss) + 1)

plt.figure('regular model loss')
plt.title('regular model loss and validation loss')
plt.plot(x_axis, training_loss, 'b', label = 'training loss')
plt.plot(x_axis, validation_loss, 'bo', label = 'validation loss')
plt.legend(loc = 'best')
plt.figure('regular model accuracy')
plt.title('regular model accuracy and validation accuracy')
plt.plot(x_axis, training_accuracy, 'b', label = 'training accuracy')
plt.plot(x_axis, validation_accuracy, 'bo', label = 'validation accuracy')
plt.legend(loc = 'best')


#fine tuning
set_trainable = False
for layer in base_convnet.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
        
    layer.trainable = set_trainable

model.compile(RMSprop(1e-5), 'binary_crossentropy', metrics = ['accuracy'])
history = model.fit(train_generator, steps_per_epoch = train_size / batch_size, epochs = 25, validation_data = validation_generator, validation_steps = validation_size / batch_size)
model.save('Chapter 5/cats and dogs classification/models/cats_and_dogs_model_vgg16_fine_tuned.h5')

history_dict = history.history
history_dict = history.history
training_loss = history_dict['loss']
validation_loss = history_dict['val_loss']
training_accuracy = history_dict['accuracy']
validation_accuracy = history_dict['val_accuracy']
x_axis = range(1, len(training_loss) + 1)

plt.figure('fine tuned model loss')
plt.title('fine tuned model loss and validation loss')
plt.plot(x_axis, training_loss, 'b', label = 'training loss')
plt.plot(x_axis, validation_loss, 'bo', label = 'validation loss')
plt.legend(loc = 'best')
plt.figure('fine tuned model accuracy')
plt.title('fine tuned model accuracy and validation accuracy')
plt.plot(x_axis, training_accuracy, 'b', label = 'training accuracy')
plt.plot(x_axis, validation_accuracy, 'bo', label = 'validation accuracy')
plt.legend(loc = 'best')
plt.show()
        