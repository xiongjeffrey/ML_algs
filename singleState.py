def getNewProb(trialMatrix, hiddenLay, k):
    #!TODO: generalize for more than 2 states
    # !NOTE: for now, k is 2

    # returns new probability vector after hidden layer is applied
    # returned probability vector has length k, where k := number of states

    # returned probability = weighted average of all trials within a given state

    # guaranteed that all values in hidden vector are integers 0 ~ k

    tempMat = []

    # get the asummed counts for each
    for i in range(len(trialMatrix)):
        a_h = hiddenLay[i][0]*trialMatrix[i][0]
        a_t = hiddenLay[i][0]*trialMatrix[i][1]

        b_h = hiddenLay[i][1]*trialMatrix[i][0]
        b_t = hiddenLay[i][1]*trialMatrix[i][1]

        tempMat.append([a_h, a_t, b_h, b_t])

    #!TODO: make this more efficient
    #print(tempMat)

    columnSums = [sum([row[i] for row in tempMat]) for i in range(0,len(tempMat[0]))]
    #print(columnSums)

    probVect = [0, 0]
    probVect[0] = columnSums[0]/(columnSums[0] + columnSums[1])
    probVect[1] = columnSums[2] / (columnSums[2] + columnSums[3])

    return probVect

def getHiddenLayer(trialMatrix, probVector):
    # !TODO: see if can generalize for more than 2 states

    # trialVector has length m*2, describing # of heads, # of tails at that trial
    # returns hiddenLayer of dimensions m*k, where m is number of trials and k is number of states
    #!NOTE: for now, k is 2

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

        #print(z)

    return z

def hasConverged(arr1, arr2): #!TODO: generalize this for more than 2 states

    epsilon = 0.001
    return abs(arr1[0] - arr2[0]) < epsilon and abs(arr1[1] - arr2[1]) < epsilon

if __name__ == '__main__':

    trialMatrix = [[5, 5], [9, 1], [8, 2], [4, 6], [7, 3]]

    states = 2
    probVector = [0.8, 0.45]
    # probVector = [1.0/states]*states # uniformly initialize an array of number of probability states

    print(probVector)

    hidLay = getHiddenLayer(trialMatrix, probVector)
    print(hidLay)

    newProb = getNewProb(trialMatrix, hidLay, states)

    print(newProb)

    while not hasConverged(probVector, newProb):

        probVector = newProb

        hidLay = getHiddenLayer(trialMatrix, probVector)
        newProb = getNewProb(trialMatrix, hidLay, states)
        print(newProb)

    # print(newProb)

#!NOTE: NOT USED
def getClosestElement(k, vect):
    # returns index of element in vect with value closest to k
    # in case of tie returns the first one

    minDist = abs(vect[0] - k)
    minIdx = 0
    for i in range(1, len(vect)):
        if abs(vect[i] - k) < minDist:
            minDist = abs(vect[i] - k)
            minIdx = i

    return minIdx