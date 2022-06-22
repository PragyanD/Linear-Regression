import numpy as np
import random
import math


def get_dataset(filename):
    """
    INPUT: 
        filename - a string representing the path to the csv file.
    RETURNS:
        An n by m+1 array, where n is # data points and m is # features.
        The labels y should be in the first column.
    """
    dataset = []
    with open(filename, 'r', encoding='UTF8') as readFile:
        next(readFile)
        for row in readFile:
            dataset.append(list(map(float, row.strip().split(',')))[1:])
    return np.array(dataset)


def print_stats(dataset, col):
    """
    INPUT: 
        dataset - the body fat n by m+1 array
        col     - the index of feature to summarize on. 
                  For example, 1 refers to density.
    RETURNS:
        None
    """
    d = dataset[:,col]
    length = len(d)
    mean = sum(d) / length
    dev = math.sqrt(sum((i - mean)**2 for i in d)/(length-1))
    print(length)
    print('{:.2f}'.format(mean))
    print('{:.2f}'.format(dev))

def regression(dataset, cols, betas):
    """
    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]
    RETURNS:
        mse of the regression model
    """
    mse = 0
    for i in range(len(dataset)):
        temp = betas[0] - dataset[:,0][i]
        for n,m in enumerate(cols):
            temp += betas[n + 1] * dataset[:,m][i]
        mse += temp ** 2
    mse = mse/len(dataset)
    return mse


def gradient_descent(dataset, cols, betas):
    """
        INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]
    RETURNS:
        An 1D array of gradients
    """
    grads = []
    grad = 0
    for i in range(len(dataset)):
        grad += betas[0] - dataset[:,0][i]
        for n,m in enumerate(cols):
            grad += betas[n+1] * dataset[:,m][i]
    grads.append(2*grad/len(dataset))
    for col in cols:
        grad = 0
        for i in range(len(dataset)):
            temp = betas[0] - dataset[:,0][i]
            for n,m in enumerate(cols):
                temp += betas[n+1] * dataset[:,m][i]
            temp *= dataset[:,col][i]
            grad += temp
        grads.append(2*grad/len(dataset))
    grads = np.array(grads)
    return grads


def iterate_gradient(dataset, cols, betas, T, eta):
    """
    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]
        T       - # iterations to run
        eta     - learning rate
    RETURNS:
        None
    """
    grads = gradient_descent(dataset,cols,betas)
    nextBetas = [b - eta*grads[j] for j,b in enumerate(betas)]
    for i in range(1,T+1):
        reg = regression(dataset,cols,nextBetas)
        print(i, '{:.2f}'.format(reg), end=' ')
        for b in nextBetas:
            print('{:.2f}'.format(b), end=' ')
        print()
        grads = gradient_descent(dataset,cols,nextBetas)
        nextBetas = [b - eta*grads[j] for j,b in enumerate(nextBetas)]

def compute_betas(dataset, cols):
    """
    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
    RETURNS:
        A tuple containing corresponding mse and several learned betas
    """
    y = dataset[:,0].reshape(len(dataset),1)
    x = [[1 for i in range(len(dataset))]]
    for col in cols:
        x.append(dataset[:,col])
    x = np.array(x).transpose()
    transpose = x.transpose()
    betas = np.dot(np.linalg.inv(np.dot(transpose,x)),np.dot(transpose,y))
    betas = betas.reshape(len(betas),)
    mse = regression(dataset,cols,betas)
    return (mse, *betas)

def predict(dataset, cols, features):
    """
    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        features- a list of observed values
    RETURNS:
        The predicted body fat percentage value
    """
    betas = compute_betas(dataset, cols)[1:]
    var = betas[0]
    for i in range(len(cols)):
        var += betas[i+1] * features[i]
    result = var
    return result

def random_index_generator(min_val, max_val, seed=42):
    """
    DO NOT MODIFY THIS FUNCTION.
    DO NOT CHANGE THE SEED.
    This generator picks a random value between min_val and max_val,
    seeded by 42.
    """
    random.seed(seed)
    while True:
        yield random.randrange(min_val, max_val)

def grad_descent_mod(dataset, cols, betas):
    """
    Modified gradient descent function for use in SGD
        INPUT: 
        dataset - a datapoint from original dataset
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]
    RETURNS:
        An 1D array of gradients
    """
    grads = []
    grad = betas[0] - dataset[0]
    for n,m in enumerate(cols):
        grad += betas[n+1] * dataset[m]
    grads.append(2*grad)
    for col in cols:
        grad = betas[0] - dataset[0]
        for n,m in enumerate(cols):
            grad += betas[n+1] * dataset[m]
        grad *= dataset[col]
        grads.append(2*grad)
    grads = np.array(grads)
    return grads

def sgd(dataset, cols, betas, T, eta):
    """
    You must use random_index_generator() to select individual data points.
    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]
        T       - # iterations to run
        eta     - learning rate
    RETURNS:
        None
    """
    rand = random_index_generator(0,len(dataset))
    for i in range(1,T+1):
        n = next(rand)
        grads = grad_descent_mod(dataset[n],cols,betas)
        betas = [b - eta*grads[j] for j,b in enumerate(betas)]
        reg = regression(dataset,cols,betas)
        print(i, '{:.2f}'.format(reg), end=' ')
        for b in betas:
            print('{:.2f}'.format(b), end=' ')
        print()