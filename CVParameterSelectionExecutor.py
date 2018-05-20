import weka.classifiers.Classifier
import weka.classifiers.Evaluation
import weka.classifiers.meta.CVParameterSelection


class CVParameterSelectionExecutor():
    def __init__(self):
        pass


    def execute(self,featureInclusion, kFold):
        deleteFeatures = 0
