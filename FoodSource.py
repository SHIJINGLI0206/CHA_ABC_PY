import numpy as np

class FoodSource():
    def __init__(self,featureAInclusion, fitness=0,nrFeatures=0):
        self.featureInclusion = featureAInclusion
        self.fitness = fitness
        self.limit = 0
        self.nrFeatures  = nrFeatures


    def getFeatureInclusion(self):
        return self.featureInclusion

    def setFeatureInclusion(self, featureInclusion):
        self.featureInclusion = featureInclusion

    def getFitness(self):
        return self.fitness

    def setFitness(self, fitness):
        self.fitness = fitness

    def getLimit(self):
        return self.limit

    def setLimit(self,limit):
        self.limit = limit

    def incrementLimit(self):
        self.limit += + 1

    def getNrFeatures(self):
        return self.nrFeatures

    def setNrFeatures(self,nrFeatures):
        self.nrFeatures = nrFeatures

    def increaseNrFeatures(self):
        self.nrFeatures += 1

    