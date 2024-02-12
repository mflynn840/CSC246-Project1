from math import floor
import os
import re
import numpy as np 
import matplotlib.pyplot as plt
from Data import Data
from sklearn.model_selection import KFold

from Data import getMinibatches
from Data import getMinibatchIdxs
from LearningRate import LearningRate

class PolyModel:

    def __init__(self, M, regularizerStrength):
        self.M = M
        self.regularizerStrength = regularizerStrength
        self.weights = np.zeros(M)
        self.initWeights = False
        self.generatorPath = ""

        #print("Model with lambda: " + str(self.regularizerStrength))

    def getDegree(self):
        return self.M - 1 #0 indexing (x^0 is counted)
    
    def fitExact(self, data: Data):

        if self.initWeights:
            inp = input("Are you sure you want to overwrite this model? (Y/N)")
            if inp == "n" or inp == "N":
                return



        X_train = data.getX(True)
        Y_train = data.getY(True)
        X_test = data.getX(False)
        Y_test = data.getY(False)

        
        return self.inverseSolver(X_train, Y_train, X_test, Y_test)
        


        

    def fitGD(self, data:Data , epochs, lr0, lrDecay, batchSize, saveLosses=True, evalFreq=1):


        x_train = data.getX()
        y_train = data.getY()

        x_test = data.getX(False)
        y_test = data.getY(False)

        return self.batchSGDSolver(x_train, y_train, x_test, y_test, epochs, lr0, lrDecay, batchSize, evalFreq=evalFreq)
    
    def getDesignMatrix(self, X):

        #print("num samples: " + str(X.shape[0]))
        #print("M: " + str(self.M))

        designMatrix = np.zeros((X.shape[0], self.M))
        for i in range(0, X.shape[0]):
            for j in range(0, self.M):
                designMatrix[i][j] = pow(X[i], j)

        return designMatrix        


    def step(self, gradient, lr):
        
        self.weights = self.weights + lr * gradient

    
    def SGDSolver(self, X_train, Y_train, X_test, Y_test, epochs:int, lr0, decayRate, evalFreq=1):
        return self.batchSGDSolver(X_train, Y_train, X_test, Y_test, epochs, lr0, decayRate, 1, evalFreq=evalFreq)


    def FullGDSolver(self, X_train, Y_train, X_test, Y_test, epochs:int, lr0, decayRate, evalFreq=1):
        return self.batchSGDSolver(X_train, Y_train, X_test, Y_test, epochs, lr0, decayRate, len(X_train), evalFreq=evalFreq)
        

    def batchSGDSolver(self, X_train, Y_train, X_test, Y_test, epochs:int, lr0, decayRate, batchSize, evalFreq=1):

        assert(X_train.shape[0] == Y_train.shape[0] and X_test.shape[0] == Y_test.shape[0])

        self.randomInitilizeWeights()

        trainLosses = []
        testLosses = []

        minibatches = getMinibatchIdxs(X_train.shape[0], batchSize)
        designMatrix = self.getDesignMatrix(X_train)


        lr = LearningRate(lr0, decayRate)

        for epoch in range(epochs):
            for idxs in minibatches:

                gradient = np.zeros((self.weights.shape[0]))

                for idx in idxs:

                    #weight update rule
                    gradient += (Y_train[idx] - self.predict(X_train[idx])) * designMatrix[idx]

                self.step(gradient, lr.getLr(epoch))
                    
            if (epoch + 1) % evalFreq == 0:
                trainLosses.append(self.getRMSLoss(X_train, Y_train))
                testLosses.append(self.getRMSLoss(X_test, Y_test))

        self.initWeights = True


        return (trainLosses, testLosses)
        
          
    def inverseSolver(self, X_train, Y_train, X_test, Y_test):

        designMatrix = self.getDesignMatrix(X_train)
        targets = Y_train

        I = np.identity(designMatrix.shape[1])
        lambdaI = self.regularizerStrength * I


        designTranspose = np.transpose(designMatrix)
        designNorm = np.matmul(designTranspose, designMatrix)

        inverse = np.linalg.pinv(np.add(lambdaI, designNorm))

        w = np.matmul(inverse, np.matmul(designTranspose, targets))
        
        self.weights = w
        self.initWeights = True


        return (self.getRMSLoss(X_train, Y_train), self.getRMSLoss(X_test, Y_test))
        


    def randomInitilizeWeights(self):
        self.weights = np.random.random((self.M))
        self.initWeights = True

    def predict(self, x):

        if self.initWeights:
            newX = np.zeros(self.M)
            for i in range(0, self.M):
                newX[i] = x**i

        return np.matmul(self.weights, newX)
    
    def predictSet(self, X):


        assert(self.initWeights)

        preds = np.zeros(X.shape[0])
        for i in range(0, X.shape[0]):
            preds[i] = self.predict(X[i])

        return preds



    def getRMSLoss(self, data, targets):

        loss = self.computeLoss(data, targets)
        return pow((2 * loss) / len(data), .5)



    def computeLoss(self, data, targets):

        assert data.shape[0] == len(targets)
        assert(self.initWeights)

        estimates = self.predictSet(data)
        loss = 0
        for i in range(0, targets.shape[0]):
            loss += (targets[i] - estimates[i]) ** 2

        return loss
    
    def __str__(self):


        string = "y(x) = "
        if not self.initWeights:
            for i in range(0, self.M):
                if i != self.M-1:
                    string += "w" + str(i) + " * x^" + str(i) + " + " 
                else:
                    string += "w" + str(i) + " * x^" + str(i)
        else:
            for i in range(0, self.M):
                if i != self.M-1:
                    string += str(round(self.weights[i], 3)) + " * x^" + str(i) + " + " 
                else:
                    string += str(round(self.weights[i], 3))  + " * x^" + str(i)
            
        string += " | Regularizer: " + str(self.regularizerStrength)
        return string


def fitKfoldExact(M, regularizer, data: Data, numFolds):

    preformanceDict = dict()
    X, Y = data.getX(True), data.getY(True)
    Kfolder = KFold(n_splits=numFolds, shuffle=False)

    for i, (trainI, testI) in enumerate(Kfolder.split(X, Y)):
        x_train, y_train = X[trainI], Y[trainI]
        x_test, y_test = X[testI], Y[testI]

        model = PolyModel(M, regularizer)
        preformanceDict.update({i : model.inverseSolver(x_train, y_train, x_test, y_test)})

    avgTrnLoss = 0
    avgTestLoss = 0

    for key, val in enumerate(preformanceDict):
        avgTrnLoss += preformanceDict[key][0]
        avgTestLoss += preformanceDict[key][1]

    avgTrnLoss, avgTestLoss = avgTrnLoss/numFolds, avgTestLoss/numFolds


    return avgTrnLoss, avgTestLoss
    



def fitKfoldGD(M, regularizer:float, data:Data, epochs:int, lr0:float, decayRate:float, numFolds:int, batchSize=1, evalFreq=None):

    if evalFreq is None:
        evalFreq = epochs #evaluate model once per fold

    if numFolds > data.size():
        print("Error: too many folds")

    
    Kfolder = KFold(n_splits=numFolds, shuffle=False)

    X, Y = data.getX(True), data.getY(True)

    preformanceDict = dict()

    for i, (trainI, testI) in enumerate(Kfolder.split(X, Y)):

        #print("Fold " + str(i) + ": " + "train=" + str(trainI) + ", " + "test=" + str(testI))

        x_train, y_train = X[trainI], Y[trainI]
        x_test, y_test = X[testI], Y[testI]

        model = PolyModel(M, regularizer)
        preformanceDict.update({i : model.batchSGDSolver(x_train, y_train, x_test, y_test, epochs, lr0, decayRate, batchSize, evalFreq=evalFreq)})



    avgTrnLoss = 0
    avgTestLoss = 0

    for key, val in enumerate(preformanceDict):
        avgTrnLoss += preformanceDict[key][0][0]
        avgTestLoss += preformanceDict[key][1][0]

    avgTrnLoss, avgTestLoss = avgTrnLoss/numFolds, avgTestLoss/numFolds

    #print(preformanceDict)

    return avgTrnLoss, avgTestLoss






           
    

