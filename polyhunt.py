from PolyModel import PolyModel

from PolyModel import fitKfoldGD
from PolyModel import fitKfoldExact
import re

from Plot import Plot
from Data import Data
from sklearn.model_selection import train_test_split
import numpy as np
import sys
import matplotlib.pyplot as plt
from tabulate import tabulate





class QuickPlot:
    def __init__(self, Xs, Ys, labels, xlab="x", ylab="y", title="Title"):

        if type(Xs) is list:
            for X, Y, label in zip(Xs, Ys, labels):
                plt.plot(X, Y, label=label, linestyle='-')
        else:
            plt.plot(X, Y)

        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.title(title)

        # Add a legend, grid, and show the plot
        plt.legend()
        plt.grid(True)
        plt.show()


class TrueFunction:
    def __init__(self, data:Data):
        self.coefficients, self.powers = self.parseFunction(data.generatorPath)
        self.degree = len(self.powers) - 1 #0 indexing (x^0 is counted)

    def evaluate(self, X):

            vals = []

            for x in X:
                vals.append(self.trueFunctionPoint(x))

            return vals

    def trueFunctionPoint(self, x):
            sum = 0

            for i in range(0,len(self.coefficients)):
                sum += self.coefficients[i] * pow(x, self.powers[i])

            return sum
    
    def printFunction(self):
        for cof, power in zip(self.coefficients, self.powers):
            if cof >= 0:
                print(" + " + str(cof) + " * x^" + str(power), end="")

            else:
                print(" " + str(cof) + " * x^" + str(power), end="")

        print("")

    def parseFunction(self, generatorPath: str):

        with open(generatorPath) as file:

            functionS = str.split(file.read(), "\n")[1]
            expression = r'\s*([-+]?\s*\d*\.?\d*)X*\^*(\s*\d*)'
            #print(functionS)
            matches = re.findall(expression, functionS)


            coefficients = []
            for i in range(len(matches)-1):
                if matches[i] != '':
                    coefficients.append(float(matches[i][0].replace(" ", "").replace("+", ""))) 

            coefficients.reverse()
            powers = [i for i in range(0, len(coefficients))]


            return (coefficients, powers) 




def regularizationTests(data, MTrue, lambdas, plotFunctions=False):

    Ms = [MTrue, 2*MTrue, 5*MTrue, 100*MTrue]

    trueFunc = TrueFunction(data)
    trueX = xTrue = np.arange(-1, 1, .001)
    trueY = trueFunc.evaluate(trueX)

    trainX = data.getX(True)
    testX = data.getX(False)
    trainY = data.getY(True)
    testY = data.getY(False)



    if plotFunctions:
        for i in range(len(Ms)):
            for j in range(len(lambdas)):
                model = PolyModel(Ms[i], lambdas[j])
                model.fitExact(data)

                modelTestY = model.predictSet(data.getX(False))
                modelTrainY = model.predictSet(data.getX(True))


                QuickPlot([trueX, trainX, trainX], [trueY, trainY, modelTrainY], ["True function", "Training data", "Model fit"], "X", "Y", "Model fit for M=" + str(Ms[i]) + " and regularizer=" + str(lambdas[j]) + " (Train data)")
                #QuickPlot([trueX, testX, testX], [trueY, testY, modelTestY], ["True function", "Testing data", "Model fit"], "X", "Y", "Model fit for M=" + str(Ms[i]) + " and regularizer=" + str(lambdas[j]) + " (Test data)")

    else:

        testLosses = np.zeros((len(Ms), len(lambdas)))
        for i in range(len(Ms)):
            for j in range(len(lambdas)):
                losses = fitKfoldExact(Ms[i], lambdas[j], data, 5)
                testLosses[i][j] = losses[1]

        

        for i in range(len(Ms)):
            losses = []
            for j in range(len(lambdas)):
                losses.append(testLosses[i][j])
            

            QuickPlot([lambdas], [losses], ["M=" + str(Ms[i])], "regularization strength", "Training RMS loss", "Loss vs. regularization for M=" + str(Ms[i]))



def sgdTests(data:Data, numFolds: int, epochs:int, dataName:str):


    #PREFORM GRID SEARCH FOR M AND LAMBDA USING EXACT SOLVERS

    #define search grid
    Ms = [i for i in range(1, 100, 5)]
    regs = [i for i in np.arange(0.1, 2.0, .1)]

    #grid1Loss[i][j] = loss of a model with degree Ms[i] and regularizer regs[j]
    grid1Loss = np.zeros((len(Ms), len(regs)))


    #Train all models and save their train set averaged loss
    for i in range(len(Ms)):
        for j in range(len(regs)):

            metrics = fitKfoldExact(Ms[i], regs[j], data, numFolds)
            grid1Loss[i][j] = metrics[1]

    
    #model selection
    optimalM = Ms[0]
    optimalReg = regs[0]
    optimalLoss = grid1Loss[0][0]

    for i in range(len(Ms)):
        for j in range(len(regs)):

            if grid1Loss[i][j] < optimalLoss:
                optimalLoss = grid1Loss[i][j]
                optimalM = Ms[i]
                optimalReg = regs[j]



    print("Best Exact model was: M=" + str(optimalM) + ", reg=" + str(optimalReg) + " | with loss=" + str(optimalLoss))


    #Search grid for sgd model
    batchSizes = [1, 75, len(data.getX())]
    lrs = [.000001, .00001, .0001, .001, .01,] #starting learning rate
    decayFactors = [0, .0001, .001, .01, .05, .1] #decay factor

    
    #grid2Loss[i][j][k] is test loss for modell with batchsizes[i] lrs[j] 
    grid2Loss = np.zeros((len(batchSizes), len(lrs), len(decayFactors)))

    #train all models and save averaged test fold loss
    for i in range(len(batchSizes)):
        for j in range(len(lrs)):
            for k in range(len(decayFactors)):

                #print("bs=" + str(batchSizes[i]) + "lr=" + str(lrs[j]) + "decay=" + str(decayFactors[k]))
                metrics = fitKfoldGD(optimalM, optimalReg, data, epochs, lrs[j], decayFactors[k], numFolds, batchSizes[i])
                grid2Loss[i][j][k] = metrics[1]
    

    #grid search to find GD model parameters (model selection 2)
    bestBatchSize = batchSizes[0]
    bestLr = lrs[0]
    bestDecayFactor = decayFactors[0]
    bestGrid2Loss = grid2Loss[0][0][0]


    for i in range(len(batchSizes)):
        for j in range(len(lrs)):
            for k in range(len(decayFactors)):
                if grid2Loss[i][j][j] < bestGrid2Loss:


                    bestGrid2Loss = grid2Loss[i][j][k]
                    bestBatchSize = batchSizes[i]
                    bestLr = lrs[j]
                    bestDecayFactor = decayFactors[k]
    
    print("Best SGD Model was: Batch size=" + str(bestBatchSize) + ", lr0=" + str(bestLr) + ", decay=" + str(bestDecayFactor) + " | with loss=" + str(bestGrid2Loss))


    #refit the optimal models and test them on unseen data
    exactModel = PolyModel(optimalM, optimalReg)
    GDModel = PolyModel(optimalM, optimalReg)

    exactLosses = exactModel.inverseSolver(data.getX(), data.getY(), data.getX(False), data.getY(False))
    GDLosses = GDModel.batchSGDSolver(data.getX(), data.getY(), data.getX(False), data.getY(False), epochs, bestLr, bestDecayFactor, bestBatchSize)


    #loss measured at each epoch (constant for exact solver)
    exactTrnLoss = [exactLosses[0] for i in range(epochs)]
    exactTstLoss = [exactLosses[1] for i in range(epochs)]
    GDTrnLoss = GDLosses[0]
    GDTstLoss = GDLosses[1]

    XAxis = np.arange(0, epochs)

    #make accuracy vs time comparison charts for optimal models (train and test data seperately)
    QuickPlot([XAxis, XAxis], [exactTstLoss, GDTstLoss], ["Matrix inversion", "Gradient Descent"], xlab="epoch", ylab="RMS loss", title="Test Set Loss vs. epoch (" + dataName + ")")
    QuickPlot([XAxis, XAxis], [exactTrnLoss, GDTrnLoss], ["Matrix inversion", "Gradient Descent"], xlab="epoch", ylab="RMS loss", title="Train Set Loss vs. epoch (" + dataName + ")")


    #get polynomial that it was generated from
    trueFunction = TrueFunction(data)
    xTrue = np.arange(-1, 1, .001)
    yTrue = trueFunction.evaluate(xTrue)

    #get train set as function
    xData = data.getX(True)
    yData = data.getY(True)

    #get model predicted ys over trainSet
    inverseModelY = exactModel.predictSet(xData)
    gdModelY = GDModel.predictSet(xData)

    
    #plot true function, train data and model fits on single graph
    QuickPlot([xTrue, xData, xData, xData], [yTrue, inverseModelY, gdModelY, yData], ["True function", "Inverse model solution", "GD model solution", "Training data"], title=dataName)




class DegreeFinder:
    def __init__(self, data):
        self.data = data

    
    def findM(self, exact=True):

        Ms = np.arange(0, 20, 1)
        #regularizers = np.arange(0, 1, 1)
        regularizers = np.arange(0, .5, .05)


        #valLlosses[i][j] is loss from model trained with M[i] and regularizers[j] as params
        valLosses = np.zeros((Ms.shape[0], regularizers.shape[0]))


        for i in range(len(Ms)):
            for j in range(len(regularizers)):

                losses = fitKfoldExact(Ms[i], regularizers[j], self.data, 5)
                valLosses[i][j] = losses[1]

        #grid search for best M and regularizer
        bestLoss = valLosses[0][0]
        bestM = Ms[0]
        bestReg = regularizers[0]

        for i in range(0, Ms.shape[0]):
            for j in range(0, regularizers.shape[0]):
                if valLosses[i][j] < bestLoss:
                    bestLoss = valLosses[i][j]
                    bestM = Ms[i]
                    bestReg = regularizers[j]
        

        
        trueFunction = TrueFunction(self.data)
        bestModel = PolyModel(bestM, bestReg)
        bestModel.inverseSolver(self.data.getX(), self.data.getY(), self.data.getX(False), self.data.getY(False))


        #print("Selected M is: " + str(bestModel.getDegree()))
        #print("Acual M is: " + str(trueFunction.degree))
        #print("Model: " + str(bestModel))
        #print("True function: ", end="")
        #trueFunction.printFunction()

        return bestM, trueFunction.degree, bestReg

        
    

def degreeFinderTest():

    datasets = []
    MEstimates = []
    MActuals = []
    regularizers = []


    noiseLevels = ["low", "moderate", "high", "clean"]
    sets = ["A", "B", "C"]

    for noiseLevel in noiseLevels:
        
        for set in sets:
            data = Data(noiseLevel, set)
            degreeFinder = DegreeFinder(data)

            results = degreeFinder.findM()
            #print("Dataset: " + noiseLevel + " " + set + " | Degree Estimate: " + str(results[0]) + "| Actual Degree: " + str(results[1]))
            MEstimates.append(results[0])
            MActuals.append(results[1])
            regularizers.append(results[2])
            datasets.append(noiseLevel + " " + set)


    tableEntries = []
    for i in range(len(datasets)):
        tableEntries.append([datasets[i], MEstimates[i], MActuals[i], regularizers[i]])

     

    print(tabulate(tableEntries, headers = ["Dataset: ", "Estimated Degree", "Actual Degree", "regularizer"], numalign="center", stralign="center", tablefmt="fancy_grid"))







degreeFinderTest()



    







#sgdTests(data, 5, 100, "High C")
#regularizationTests(data, 6, [0, 0.001, 0.01, 0.1, 0.5, 1, 5, 10])












#print(fitKfoldGD(10, 1, data, 1000, .001, 0, 5, 10))
#sgdStats()
    
#regularizationStats(10, np.arange(0, .00001, .000001))
#0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 10000


#plotter = Plot(data, model)
#plotter.plot()




#finder = DegreeFinder(trainData)
#finder.findM()

#X, Y = trainData.getFullDataset()


#model.fit(X, Y)
#print(model)
#preds = model.predictSet(X)
#print("loss: " + str(model.computeLoss(Y, preds)))
#Plot(trainData.generatorPath, X, Y, preds)

