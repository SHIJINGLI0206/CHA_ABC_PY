import FoodSource
import io
from enum import Enum
import PerturbationStrategy
import numpy as np
import random
import math
from datetime import datetime

class FeatureSelection():
    def __init__(self,runtime, limit, mr):
        self.KFOLD = 10
        self.featureSize = 0
        self.limit = limit
        self.runtime = runtime
        self.bestFitness = 0
        self.bestFoodSource = None

        self.mr = mr
        self.foodSources = set()
        self.visitedFoodSources = set()
        self.scouts = set()
        self.abandoned = set()
        self.markedToRemoved = set()
        self.neighbors  = set()
        if mr > 0:
            self.perturbation = PerturbationStrategy.USE_MR
        else:
            self.perturbation = PerturbationStrategy.CHANGE_ONE_FEATURE

        self.states = 0



    def execute(self):
        self.visitedFoodSources = set()
        self.states = 0
        self.logFeatureSelectionInit(self.runtime,self.limit,self.mr,self.perturbation,0)
        time = datetime.now()
        self.initializeFoodSource()
        print('init time: ',datetime.now() - time)
        for i in range(0,self.runtime):
            self.sendEmployedBees()
            self.sendOnlookerBees()
            self.sendScoutBeesAndRemoveAbandonsFoodSource()

        time = (datetime.now() - time) / 60000
        self.states = 0






    def initializeFoodSource(self):
        print('initializeFoodSources')
        for i in range(0,self.featureSize):
            self.states += 1
            features = np.zeros(self.featureSize)
            features[i] = True
            curFitness = self.calculateFitness(features)
            fs = FoodSource(features,curFitness,1)
            self.foodSources.add(fs)
            if(curFitness >  self.bestFitness):
                self.bestFoodSource = fs
                self.bestFitness = curFitness


    def sendEmployedBees(self):
        print('sendEmployedBees')
        self.scouts = set()
        self.markedToRemoved = set()
        self.neighbors = set()

        for fs in self.foodSources:
            self.sendBee(fs)

        # remove all markedToRemoved
        for n in self.neighbors:
            self.foodSources.add(n)


    def sendOnlookerBees(self):
        print('SendOnlookerBees')
        self.markedToRemoved = set()
        self.neighbors = set()

        min = 0
        range = 0
        for s in self.foodSources:
            if s.getFitness() < min:
                min = s.getFitness()
            if s.getFitness() > range:
                range = s.getFitness()

        for fs in self.foodSources:
            prob = (fs.getFitness()-min)/range
            if random.uniform(0,1) < prob:
                self.sendBee(fs)
            else:
                fs.incrementLimit()

        self.foodSources.clear()
        for n in self.neighbors:
            self.foodSources.add(n)



    def sendBee(self,foodSource):
        features = foodSource.getFeatureInclusion()
        nrFeatures = foodSource.getNrFeatures()
        times = 0
        modifedFoodSource = None
        while 1:
            times += 1
            if self.perturbation == PerturbationStrategy.PerturbationStrategy.CHANGE_ONE_FEATURE:
                index = round(random() * (self.featureSize - 1))
                if features[index] is False:
                    nrFeatures += 1
                    features[index] = True
            elif self.perturbation == PerturbationStrategy.PerturbationStrategy.USE_MR:
                for i in range(0,self.featureSize):
                    if random() < self.mr:
                        if features[i] == False:
                            nrFeatures += 1
                            features[i] = True

            modifedFoodSource = FoodSource(features)
            if modifedFoodSource not in self.foodSources and \
                            modifedFoodSource not in self.neighbors and \
                            modifedFoodSource not in self.abandoned and \
                            modifedFoodSource not in self.visitedFoodSources and \
                            times > self.featureSize:
                break

        if modifedFoodSource not in self.foodSources or \
            modifedFoodSource not in self.neighbors or \
            modifedFoodSource not in self.visitedFoodSources or \
            modifedFoodSource not in self.abandoned:
            self.states += 1
            fitness = self.calculateFitness(features)
            modifedFoodSource.setFitness(fitness)
            modifedFoodSource.setNrFeature(nrFeatures)
            if foodSource.getFitness() > fitness or \
                    (fitness == foodSource.getFitness() and nrFeatures > foodSource.getNrFeatures()):
                foodSource.incrementLimit()
                if foodSource.getLimit() >= self.limit:
                    self.markAbandonsFoodSource(foodSource)
                    self.createScoutBee()
                    self.visitedFoodSources.add(modifedFoodSource)
                self.visitedFoodSources.add(modifedFoodSource)
            else:
                if fitness > self.bestFitness or (fitness == self.bestFitness and nrFeatures < self.bestFoodSource.getNrFeatures()):
                    self.bestFoodSource = FoodSource(modifedFoodSource)
                    self.bestFitness = fitness
                self.neighbors.add(modifedFoodSource)
        return True




    def createScoutBee(self):
        features = np.array(len(self.featureSize))
        foodSource = None
        nrFeatures = 0
        for j in range(0,self.featureSize):
            inclusio = random.choice([True,False])
            if inclusio:
                nrFeatures += 1
            features[j] = inclusio


        curFitness = self.calculateFitness(features)
        foodSource = FoodSource(features,curFitness,nrFeatures)
        if foodSource not in self.foodSources or \
                        foodSource not in self.neighbors or \
                        foodSource not in self.abandoned or \
                        foodSource not in self.visitedFoodSources:
            self.states += 1
            self.scouts.add(foodSource)

    def sendScoutBeesAndRemoveAbandonsFoodSource(self):
        self.foodSources.clear()
        for s in self.scouts:
            self.foodSources.add(s)

    def markAbandonsFoodSource(self,foodSource):
        self.abandoned.add(foodSource)

    def calculateFitness(self,features):
        pass

    def setExecutor(self,executor):
        pass




    def logFeatureSelectionInit(self,runtime,limit, mr, perturbation,nrFeature):
        print('Feature Selection START --------')
        print("Runtime [" + runtime + "], Limit [" + limit +
              "], MR [" + mr + "], perturbation [" + perturbation + "]" )


    def logBestSolutionAndExecutionTime(self,t):
        print("Best bestFoodSource" )
        print("Executedo em " + t + " percorrendo " + self.states + " states ")
        print("Feature Selection END -------")




