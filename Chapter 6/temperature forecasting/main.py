import numpy as np, os
import keras
from matplotlib import pyplot as plt
from keras.layers import Flatten, Dense, LSTM, Input, GRU, Bidirectional, Dropout, Conv1D, MaxPooling1D
from keras.optimizers import RMSprop, Adam
from keras.losses import MeanSquaredError
from matplotlib import pyplot as plt


with open('Chapter 6/temperature forecasting/jena_climate_2009_2016.csv') as data_file:
    data = data_file.read()
    
lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

float_data = np.zeros((len(lines), len(header) - 1))

for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] =  values

training_timesteps = 200000

mean = float_data[:training_timesteps].mean(axis = 0)
float_data -= mean
std = float_data[:training_timesteps].std(axis = 0)
float_data /= std

def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
        
    i = min_index + lookback
    while True:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        
        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = list(range(row - lookback, row, step))
            samples[j] = data[indices]
            targets[j] = data[row + delay][1]
        yield samples, targets


lookback = 1440 # 10 days
step = 3
delay = 144 # one day
batch_size = 128
epochs = 20

train_generator = generator(float_data, lookback = lookback, delay = delay, 
                            min_index = 0, max_index = 200000, 
                            shuffle = True, step = step, 
                            batch_size = batch_size)

validation_generator = generator(float_data, lookback = lookback, delay = delay, 
                            min_index = 200001, max_index = 300000, 
                            step = step, 
                            batch_size = batch_size)

test_generator = generator(float_data, lookback = lookback, delay = delay, 
                            min_index = 300001, max_index = None, 
                            step = step, 
                            batch_size = batch_size)

training_steps = (200000 - lookback)//batch_size
validation_steps = (300000 - 200001 - lookback)//batch_size
test_steps = (len(float_data) - 300001 - lookback)//batch_size


model = keras.models.Sequential([
    Conv1D(32, 5, activation = 'relu', input_shape = (lookback//step, float_data.shape[-1])),
    MaxPooling1D(3),
    Conv1D(32, 5, activation = 'relu'),
    GRU(32, dropout = 0.1),
    Dropout(0.3),
    Dense(1)
])

model.summary()
model.compile(RMSprop(learning_rate = 0.001), 'mae', metrics = ['mae'])
history = model.fit(train_generator, batch_size = batch_size, epochs = epochs, 
          validation_data = validation_generator, steps_per_epoch = 500, 
          validation_steps = validation_steps)

model.evaluate(test_generator, steps = test_steps)

history_dict = history.history
loss = history_dict['loss']
validation_loss = history_dict['val_loss']
epochs_range = range(1, epochs + 1)

plt.figure('validation and training MAE')
plt.plot(epochs_range, loss, 'bo', label = 'training MAE')
plt.plot(epochs_range, validation_loss, 'b', label = 'validation MAE')
plt.legend(loc = 'best')
plt.plot()
plt.show()
