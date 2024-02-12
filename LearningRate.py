class LearningRate:

    def __init__(self, lr0, decayRate):
        self.lr0 = lr0
        self.decayRate = decayRate
    
    def getLr(self, epoch):
        return pow((1-self.decayRate), epoch) * self.lr0
