import javabridge
from weka.classifiers import Classifier
from FeatureSelectionExecution import FeatureSelectionExecution


class WineFeatureSelectionExecution(FeatureSelectionExecution):
    def __init__(self, features):
        FeatureSelectionExecution.__init__()
        self.databaseName = "wine.arff"
        self.features = features
        self.runtime = 50
        self.limit = 10
        self.mr = 0.01
        self.ibk = Classifier(classname="Lweka/classifiers.lazy.IBK",ckargs={'-K':1})


    def executeAll(self):
        self.executeFullFeaturesWithNoFilters()
        self.executeWithNoFilter()



if __name__ == '__main__':
    features = {True , True , True , True , True , True , True , True ,
                True , True , True , True , True , True , True , True , True , True ,
                True , True , True , True , True , True , True}
    fs = WineFeatureSelectionExecution(features)
    fs.executeAll()