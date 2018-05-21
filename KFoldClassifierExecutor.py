from weka.core.classes import Random
from weka.classifiers import Evaluation
from weka.classifiers import Classifier
from ClassifierExecutor import ClassifierExecutor
import javabridge

class KFoldClassifierExecutor(ClassifierExecutor):
    def __init__(self,classifier):
        ClassifierExecutor.__init__()
        self.classifier = classifier


    def execute(self,featureInclusion, kFold):
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


    def execute(self,featureInclusion, kFold, classIndex):
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

