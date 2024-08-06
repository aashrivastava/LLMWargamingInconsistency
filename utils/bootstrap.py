import numpy as np

def bootstrap_mean(dataset: list[float], n: int=1000):
    '''
    Given a dataset of inconsistency metrics, boostrap to get uncertainty in estimator

    Input:
        dataset: list[float]
            collection of inconsistency metrics to sample from
    '''
    means = []
    for i in range(n):
        curr_dataset = []
        for j in range(len(dataset)):
            curr_dataset.append(dataset[np.random.randint(0, len(dataset))])
        means.append(np.mean(curr_dataset))
    
    return means

def bootstrap_percent_over(dataset: list[float], baseline: float, n: int=1000) -> list[float]:
    '''
    Given a dataset of pairs, bootstrap to get confidence on percent over/under

    Input:
        dataset: list[float]
            collection of pairs to sample from
    '''
    percents_over = []
    for i in range(n):
        curr_dataset = np.zeros(len(dataset))
        for j in range(len(dataset)):
            curr_dataset[j] = dataset[np.random.randint(0, len(dataset))]
        percents_over.append(np.sum(curr_dataset > baseline) / 15200)
    
    return percents_over



'''
PSEUDOCODE:
means = new list
for i=1,...1_000:
    curr_dataset = new list
    for j =1,..., length of dataset:
        randomly sample a value from original dataset and add to curr_dataset (do not remove from original dataset)
    calculate mean of curr_dataset
    add mean to means list

'''

