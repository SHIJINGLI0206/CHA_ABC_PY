import javabridge
from weka.classifiers import Classifier
from FeatureSelectionExecution import FeatureSelectionExecution


class IrisFeatureSelectionExecution(FeatureSelectionExecution):
    def __init__(self, features):

        self.databaseName = "iris.arff"
        self.features = features
        self.runtime = 20
        self.limit = 6
        self.mr = 0.1
        FeatureSelectionExecution.__init__(self,self.databaseName,self.features,self.runtime,
                                           self.limit,self.mr)


    def executeAll(self):
        self.executeFullFeaturesWithNoFilters()
        self.executeWithNoFilter()



if __name__ == '__main__':
    features = {True , True , True , True }
    fs = IrisFeatureSelectionExecution(features)
    fs.executeAll()