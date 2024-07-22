import random
import numpy as np

def bootstrap(dataset: list[float], n: int=1000):
    '''
    Given a dataset of inconsistency metrics, boostrap to get a uncertainty in estimator

    Input:
        dataset: list[float]
            collection of inconsistency metrics to sample from
    '''
    means = []
    for i in range(n):
        curr_dataset = []
        for j in range(len(dataset)):
            curr_dataset.append(dataset[random.randint(0, len(dataset) - 1)])
        means.append(np.mean(curr_dataset))
    
    return means

'''
PSEUDOCODE:
means = new list
for i=1,...1_000:
    curr_dataset = new list
    for j in range 1,..., length of dataset:
        randomly sample a value from original dataset and add to curr_dataset with replacement
    calculate mean of curr_dataset
    add mean to means list

plot a histogram of the values in means list

'''