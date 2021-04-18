from utility.proxy.proxy import Proxy
from dataset.fashion_mnist import FashionMNIST, FashionMNISTDataset
from dataset.lfpeople import lfwPeople, lfwPeopleDataset

class Loader(object):

    def __init__(self):
        self._allDatasets = {}
        self.registerAllDatasets()

    def registerAllDatasets(self):
        print("Registering : fashion_mnist")
        self._allDatasets['fashion_mnist']  = Proxy(FashionMNIST(FashionMNISTDataset()))
        print("Registering : lfwp")
        self._allDatasets['lfwp']           = Proxy(lfwPeople(lfwPeopleDataset(60)))

    def showDatasets(self):
        print("Avaiable Datasets List")
        for index, key in enumerate(self._allDatasets.keys()):
            print(str(index) + ". " + str(key))
    
    def load(self, name, params = None):
        if name not in self._allDatasets.keys():
            raise Exception("Dataset is not registered")

        return self._allDatasets[name].request(params)
