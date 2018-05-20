from weka.core.dataset import Instances
from  weka.filters import Filter, MultiFilter, StringToWordVector
from weka.core.dataset import Attribute, Instance
from random import randint
from weka.core.converters import Loader,load_any_file

class ClassifierExecutor():
    def __init__(self):
        self.originalInstances = None
        self.instances = None

    def loadFeatures(self,filename,filter):
        loader = Loader("weka.core.converters.ArffLoader")
        data = loader.load_file(filename)
        self.originalInstances = data

        if filter is not None:
            for i in range(0,filter.length):
                filter[i].setInputFormat(self.originalInstances)

                #self.originalInstances = Filter.


