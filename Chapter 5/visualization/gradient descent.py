import numpy as np
from matplotlib import pyplot as plt

def loss_function(x): #Think of this as the loss function
    return x**4 - 8*x**2 + 16


def derivative(x, h = 1e-12):
    '''
    The general formula for the derivative of any function is:
    d = (f(x + h) - f(x)) / ((x + h) - (x))
    '''
    
    return (loss_function(x + h) - loss_function(x)) / (h)


def gradient_descent(initial_point):

    plt.figure('Loss function')
    learning_rate = 1e-5
    current_point = initial_point
    epochs = 10000
    time_to_finish = 10
    timeout = time_to_finish / epochs
    samples = 10000
    
    for i in range(epochs):  
        x_axis = np.array(range(int(current_point - samples/2), int(current_point + samples/2 + 1))) * 10/samples
        y_axis = loss_function(x_axis)
        current_loss = loss_function(current_point)
        
        plt.plot(x_axis, y_axis)
        plt.plot(current_point, current_loss, 'bo', label = f'current point coordinates: {(round(current_point, 5), round(current_loss, 5))}')
        plt.legend(loc = 'best')
        
        if not plt.fignum_exists('Loss function'):
            break
        
        if (i + 1) < epochs:
            plt.draw()
            plt.waitforbuttonpress(timeout)
            plt.clf()
        else:
            plt.show()

        current_point -= derivative(current_point) * learning_rate
                
gradient_descent(initial_point = 5)