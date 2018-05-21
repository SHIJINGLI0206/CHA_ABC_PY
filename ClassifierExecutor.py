from weka.core.dataset import Instances
from  weka.filters import Filter, MultiFilter, StringToWordVector
from weka.core.dataset import Attribute, Instance
from random import randint
from weka.core.converters import Loader,load_any_file
import javabridge
import weka.core.jvm as jvm
from abc import ABC, abstractmethod



class ClassifierExecutor(ABC):
    def __init__(self):
        self.originalInstances = None
        self.instances = None

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

    def loadFeatures(self):
        self.instances = self.originalInstances

    def getFeaturesSize(self):
        if self.originalInstances is None:
            return -1
        return self.originalInstances.num_attributes() - 1

    @classmethod
    @abstractmethod
    def execute(self,featureInclusion, k):
        pass

    @classmethod
    @abstractmethod
    def execute(self,featureInclusion,kFold,classIndex):
        pass


