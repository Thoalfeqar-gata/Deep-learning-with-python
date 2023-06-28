import numpy as np
from keras.datasets import boston_housing
from matplotlib import pyplot as plt
from keras import models, layers

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
mean = train_data.mean(axis = 0)
std = train_data.std(axis = 0)
train_data -= mean
std = train_data.std(axis = 0)
train_data /= std

test_data -= mean
test_data /= std

def build_model():
    model = models.Sequential([
        layers.Dense(64, 'relu'),
        layers.Dense(64, 'relu'),
        layers.Dense(1, 'linear')
    ])
    
    model.compile('rmsprop', loss = 'mse', metrics = ['mae'])
    return model

def smooth_curve(points, factor = 0.9):
    smoothed_points = [points[0]]
    for point in points[1:]:
        previous = smoothed_points[-1]
        smoothed_points.append(previous * factor + point * (1-factor))
    return smoothed_points
    
    
k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []
all_mae_histories = []

for i in range(k):
    print(f'Processing fold {i}')
    val_data = train_data[i * num_val_samples : (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples : (i + 1) * num_val_samples]
    
    partial_training_data = np.concatenate([
        train_data[:i * num_val_samples],
        train_data[(i + 1) * num_val_samples:]
    ], axis = 0)
    
    partial_training_targets = np.concatenate([
        train_targets[:i * num_val_samples],
        train_targets[(i + 1) * num_val_samples:]
    ], axis = 0)
    
    model = build_model()
    history = model.fit(partial_training_data, partial_training_targets, validation_data = (val_data, val_targets), epochs = num_epochs, batch_size = 1, verbose = 0)
    mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history)
    
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose = 0)
    all_scores.append(val_mae)
average_mae_history = np.mean(np.array(all_mae_histories), axis = 0)

plt.figure()
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')

smooth_mae_history = smooth_curve(average_mae_history[10:])
plt.figure()
plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

