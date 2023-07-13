import numpy as np

def reweight_distribution2(original_distribution, temperature):
    distribution = original_distribution ** (1/temperature)
    return distribution / np.sum(distribution)


