from weka.classifiers import Classifier
from weka.filters import Filter,MultiFilter
from FeatureSelection import FeatureSelection


class FeatureSelectionExecution:
    def __init__(self, databaseName, features, runtime, limit, mr, classifier):
        self.KFOLD = 10
        self.replaceMissingValues = None
        self.zscore = 0
        self.normalize = 0
        self.executor = None
        self.featureSelection = None
        self.features = features
        self.runtime = runtime
        self.limit = limit
        self.mr = mr
        self.databaseName = databaseName


    def executeAll(self):
        self.executeFullFeaturesWithNoFilters()
        self.executeWithNoFilter()
        self.executeFullFeaturesNormalized()
        self.executeWithNormalization()
        self.executeFullFeaturesZScore()
        self.executeWithZScore()

    def executeWithNoFilter(self):
        print('executeWithNoFilter')
        self.executor.loadFeatures(self.databaseName,self.replaceMissingValues)
        self.featureSelection = FeatureSelection(self.runtime,
                                                 self.limit,self.mr,self.executor)
        self.featureSelection.setExecutor(self.executor)
        self.featureSelection.execute()


    def executeWithNormalization(self):
        self.executor.loadFeatures(self.databaseName,self.replaceMissingValues,self.normalize)
        self.featureSelection = FeatureSelection(self.runtime,self.limit,self.mr,self.executor)
        self.featureSelection.setExecutor(self.executor)
        self.featureSelection.execute()

    def executeWithZScore(self):
        print('executeWithZScore')
        self.executor.loadFeatures(self.databaseName,self.replaceMissingValues, self.zscore)
        self.featureSelection = FeatureSelection(self.runtime,self.limit,self.mr,self.executor)
        self.featureSelection.setExecutor(self.executor)
        self.featureSelection.execute()

    def executeFullFeaturesWithNoFilters(self):
        print('executeFullFeaturesWithNoFilters')
        self.executor.loadFeatures(self.databaseName,self.replaceMissingValues)
        result = self.executor.execute(self.features,self.KFOLD)
        print('Full '+result+'%')

    def executeFullFeaturesNormalized(self):
        print('executeFullFeaturesNormalized')
        self.executor.loadFeatures(self.databaseName,self.replaceMissingValues,self.normalize)
        result = self.executor.execute(self.features,self.KFOLD)
        print('Full '+result + '%')

    def executeFullFeaturesZScore(self):
        print('executeFullFeaturesZScore')
        self.executor.loadFeatures(self.databaseName,self.replaceMissingValues,self.zscore)
        result = self.executor.execute(self.features,self.zscore)
        result = self.executor.execute(self.features,self.KFOLD)
        print('Full ' + result + '%')

    def setDatabaseName(self,databaseName):
        self.databaseName = databaseName








