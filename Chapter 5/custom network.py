import os, shutil, numpy as np, random, keras, cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

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
batch_size = 64
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
    
    
model = Sequential([
    Conv2D(32, (3, 3), activation = 'relu', input_shape = (200, 200, 3)),
    MaxPooling2D(),
    Conv2D(64, (3, 3), activation = 'relu'),
    MaxPooling2D(),
    Conv2D(64, (3, 3), activation = 'relu'),
    MaxPooling2D(),
    Conv2D(128, (3, 3), activation = 'relu'),
    MaxPooling2D(),
    Conv2D(128, (3, 3), activation = 'relu'),
    MaxPooling2D(),
    Flatten(),
    Dropout(0.3),
    Dense(512, activation = 'relu'),
    Dropout(0.3),
    Dense(64, activation = 'relu'),
    Dense(1, 'sigmoid')
])