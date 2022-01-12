import numpy as np

def getProb(trialMatrix, probVector):
    # !TODO: see if can generalize for more than 2 states

    # trialVector has length m*2, describing # of heads, # of tails at that trial
    # returns new probability vector

    # for each trial, find closest element of probVector
    z = []

    for i in range(len(trialMatrix)):

        if probVector[0] == 0:
            proportion = 0

        elif probVector[1] == 0:
            proportion = 1

        else:
            proportion = (probVector[0]/probVector[1])**trialMatrix[i][0]
            proportion = proportion*(((1 - probVector[0])/(1 - probVector[1])))**trialMatrix[i][1]

        b = 1/(1 + proportion)
        a = 1 - b

        z.append([a, b])

    # z is hidden layer of dimensions m*k, where m is number of trials and k is number of states
    # !NOTE: for now, k is 2
    z = np.transpose(z)
    print(z)

    # matrix multiplication
    # matrix returned: each row is a condition (e.g. Coin A, Coin B) for which probabilities are generated
    #                   each column is a state (e.g. tails, heads) for which each condition can exist in
    #                   each element is the probability that that condition adopts that state
    #                   sum of row adds up to 1 (total probabilities of all possible states per condition)

    proportionMatrix = np.matmul(z, trialMatrix)
    print(proportionMatrix)

    newProb = [0]*len(proportionMatrix)

    #!TODO: generalize this beyond binary, if possible
    for i in range(len(newProb)):
        newProb[i] = proportionMatrix[i][0]/sum(proportionMatrix[i])

    return newProb

def hasConverged(arr1, arr2): #!TODO: generalize this for more than 2 states

    epsilon = 0.001
    return abs(arr1[0] - arr2[0]) < epsilon and abs(arr1[1] - arr2[1]) < epsilon

if __name__ == '__main__':

    trialMatrix = [[5, 5], [9, 1], [8, 2], [4, 6], [7, 3]]

    states = 2
    probVector = [0.6, 0.5]
    # probVector = [1.0/states]*states # uniformly initialize an array of number of probability states

    print(probVector)

    newProb = getProb(trialMatrix, probVector)

    print(newProb)

    while not hasConverged(probVector, newProb):

        probVector = newProb

        newProb = getProb(trialMatrix, probVector)
        print(newProb)
