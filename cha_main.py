import javabridge
from datetime import datetime
from weka.classifiers import Classifier,Evaluation
from weka.filters import Filter,MultiFilter
from weka.core.dataset import Instances
from  weka.filters import Filter, MultiFilter, StringToWordVector
from weka.core.dataset import Attribute, Instance
from random import randint
from weka.core.converters import Loader,load_any_file
import javabridge
import weka.core.jvm as jvm
from abc import ABC, abstractmethod
from enum import Enum
from FoodSource import FoodSource
from scipy.io import arff
from io import StringIO
from weka.core.classes import Random
import numpy as np
import random

class PerturbationStrategy(Enum):
    USE_MR = 1
    CHANGE_ONE_FEATURE = 2


class CHA():
    def __init__(self):
        self.features = {True, True, True, True}
        self.databaseName = "../dataset/iris.arff"
        self.runtime = 20
        self.limit = 6
        self.mr = 0.1
        self.KFOLD = 10

        self.bestFitness = 0
        self.bestFoodSource = None
        self.foodSources = set()
        self.visitedFoodSources = set()
        self.scouts = set()
        self.abandoned = set()
        self.markedToRemoved = set()
        self.neighbors  = set()
        if self.mr > 0:
            self.perturbation = PerturbationStrategy.USE_MR
        else:
            self.perturbation = PerturbationStrategy.CHANGE_ONE_FEATURE

        self.states = 0


    def loadFeatures(self,filename,filter):
        loader = Loader("weka.core.converters.ArffLoader")
        data = loader.load_file(filename)
        self.originalInstances = data
        if filter:
            for i in range(0,filter.length):
                filter[i].setInputFormat(self.originalInstances)

                self.originalInstances = Instance(javabridge.static_call(
                    "Lweka/filters/Filter;", "useFilter",
                    "(Lweka/core/Instances;Lweka/filters/Filter;)Lweka/core/Instances;",
                    self.originalInstances,filter[i]
                ))
        self.instances = self.originalInstances
        return self.originalInstances.num_attributes() - 1


    def loadFeatures(self,filename):
        f = Filter()
        return self.loadFeatures(filename,f)

        if 0:
            with open(self.databaseName) as my_file:
                data = my_file.read()
                f = StringIO(data)
                d, meta = arff.loadarff(f)

    def loadFeatures(self):
        #self.instances = self.originalInstances
        loader = Loader("weka.core.converters.ArffLoader")
        data = loader.load_file(self.databaseName)
        self.originalInstances = data
        self.instances = Instances.copy_instances(self.originalInstances)
        return self.originalInstances.num_attributes() - 1


    def getFeaturesSize(self):
        if self.originalInstances is None:
            return -1
        return self.originalInstances.num_attributes() - 1

    def executeKFoldClassifier(self,featureInclusion, kFold):
        deleteFeatures = 0
        for i in range(0,len(featureInclusion)):
            if featureInclusion[i]:
                self.instances.deleteAttributeAt(i - deleteFeatures)
                deleteFeatures += 1
        self.instances.setClassIndex(self.instances.numAttributes() - 1)

        cvParameterSelection = javabridge.make_instance("Lweka/classifiers/meta/CVParameterSelection","()V")
        javabridge.call(cvParameterSelection, "setNumFolds", "(I)V", kFold)
        javabridge.call(cvParameterSelection,"buildClassifier(Lweka/core/Instances)V",self.instances)


        eval = Evaluation(self.instances)
        eval.crossvalidate_model(cvParameterSelection,self.instances,kFold,Random(1))

        return eval.percent_correct()


    def executeKFoldClassifier(self,featureInclusion, kFold, classIndex):
        deletedFeatures = 0
        for i in range(0,len(featureInclusion)):
            if featureInclusion[i] == False:
                self.instances.deleteAttributeAt( i - deletedFeatures)
                deletedFeatures += 1

        self.instances.setClassIndex(classIndex)

        cvParameterSelection = javabridge.make_instance("Lweka/classifiers/meta/CVParameterSelection","()V")
        javabridge.call(cvParameterSelection, "setNumFolds", "(I)V", kFold)
        javabridge.call(cvParameterSelection,"buildClassifier(Lweka/core/Instances)V",self.instances)

        eval = Evaluation(self.instances)
        eval.crossvalidate_model(cvParameterSelection, self.instances, kFold, Random(1))

        return eval.percent_correct()



    def executeFullFeaturesWithNoFilters(self):
        print('executeFullFeaturesWithNoFilters')
        self.executor.loadFeatures(self.databaseName, self.replaceMissingValues)
        result = self.executor.execute(self.features, self.KFOLD)
        print('Full ' + result + '%')

    def executeWithNoFilter(self):
        print('executeWithNoFilter')
        self.executor.loadFeatures(self.databaseName, self.replaceMissingValues)
        # self.featureSelection = FeatureSelection(self.runtime,
        #                                          self.limit, self.mr, self.executor)
        # self.featureSelection.setExecutor(self.executor)
        # self.featureSelection.execute()
        self.executeFeatureSelection()


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
            if randint(0,1) < prob:
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
                index = round(Random(1) * (self.featureSize - 1))
                if features[index] is False:
                    nrFeatures += 1
                    features[index] = True
            elif self.perturbation == PerturbationStrategy.PerturbationStrategy.USE_MR:
                for i in range(0,self.featureSize):
                    if Random(1) < self.mr:
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
            inclusio = random.r.choice([True,False])
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

    def executeFeatureSelection(self):
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

    def runCHA(self):
        self.loadFeatures()
        self.executeKFoldClassifier(self.features,self.KFOLD)
        self.executeFeatureSelection()




if __name__ == '__main__':
    print('******************************')
    print('[%s] : Start' % datetime.now())
    print('******************************')
    cha = CHA()
    cha.runCHA()
    print('******************************')
    print('[%s] : End' % datetime.now())
    print('******************************')





