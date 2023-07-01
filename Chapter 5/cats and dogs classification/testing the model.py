import cv2, numpy as np, os, keras

testing_data = 'Chapter 5/cats and dogs classification/data/test'
image_size = (200, 200)
model = keras.models.load_model('Chapter 5/cats and dogs classification/models/cats_and_dogs_model.h5')

for filename in os.listdir(testing_data):
    image = cv2.imread(os.path.join(testing_data, filename))
    input_image = cv2.resize(image, image_size) / 255.0
    prediction = model.predict(np.expand_dims(input_image, axis = 0))[0][0]
    if prediction >= 0.5:
        name = 'dog'
    else:
        name = 'cat'
        
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyWindow(name)
    