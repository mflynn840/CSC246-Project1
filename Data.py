import os
import re
import numpy as np 
import matplotlib.pyplot as plt


class Data:

    def __init__(self, noiseLevel, setName):


        noiseLevels = ["low", "moderate", "high", "clean"]
        sets = ["A", "B", "C"]
        subsets = ["test", "train", "generator"]

        if noiseLevel not in noiseLevels or setName not in sets:
            print("Error: Invalid selection")
            return None
        

        self.dataFolderPath = os.path.join(os.getcwd(), "data", noiseLevel, setName)
        self.generatorPath = os.path.join(os.getcwd(), "data", noiseLevel, setName, "generator.txt")


        assert(len(self.getX(True)) == len(self.getY(True)))
        assert(len(self.getX(False)) == len(self.getY(False)))


        self.trainSize = len(self.getX(True))
        self.testSize = len(self.getX(False))

        

    def size(self, trainSet=True):
        return self.trainSize if trainSet else self.testSize

    def getFullDataset(self, trainSet: bool):
        return self.load(trainSet)
    
    def getX(self, trainSet=True):
        return self.load(trainSet)[0]
    
    def getY(self, trainSet=True):
        return self.load(trainSet)[1]

    def getPath(self):
        return self.dataPath
    
    def getMinibatches(self, batchSize, trainSet=True):
        if batchSize > self.size(trainSet):
            print("Error: minibatch size larger than dataset size")

        numBatches = self.size(trainSet) // batchSize 
        
        X = self.getX(trainSet)
        Y = self.getY(trainSet)

        minibatches = []

        for i in range(numBatches):
            minibatch = (X[i*batchSize : ((i + 1) * batchSize)], Y[i*batchSize : ((i + 1) * batchSize)])
            minibatches.append(minibatch)

        #leftovers
        if self.size() % batchSize != 0:
            minibatch.append((X[numBatches * batchSize:], Y[numBatches * batchSize:]))
        
        return minibatches

    def load(self, trainSet=True):

        if trainSet:
            filePath = os.path.join(self.dataFolderPath, "train.txt")
        else:
            filePath = os.path.join(self.dataFolderPath, "test.txt")

        with open(filePath, 'r+') as file:

                pattern = r"(-?\d+\.\d+)\s+(-?\d+\.\d+)"
                # Find all matches in the data using the pattern
                matches = re.findall(pattern, file.read())
                X = [float(match[0]) for match in matches]
                Y = [float(match[1]) for match in matches]
                return (np.array(X), np.array(Y))
            
    def __str__(self):

        string = ""
        for i in range(0, self.n):
            string += "(" + str(self.X[i]) + ", " + str(self.Y[i]) + ")" + "\n"

        return string
    



def getMinibatchIdxs(dataSize, batchSize):
    fullIdx = np.arange(0, dataSize)

    numBatches = dataSize // batchSize 

    minibatches = []
    
    for i in range(numBatches):
        minibatchIdxs = fullIdx[i*batchSize : ((i + 1) * batchSize)]
        minibatches.append(minibatchIdxs)

    if dataSize % batchSize != 0:
        minibatches.append(fullIdx[numBatches * batchSize:])

    return minibatches
    

            
def getMinibatches(X, Y, batchSize):
    if batchSize > X.shape[0]:
        print("Error: minibatch size larger than dataset size")

    numBatches = X.shape[0] // batchSize 
    

    minibatches = []

    for i in range(numBatches):
        minibatch = (X[i*batchSize : ((i + 1) * batchSize)], Y[i*batchSize : ((i + 1) * batchSize)])
        minibatches.append(minibatch)

    #leftovers
    if X.shape[0] % batchSize != 0:
        minibatch.append((X[numBatches * batchSize:], Y[numBatches * batchSize:]))
    
    return minibatches