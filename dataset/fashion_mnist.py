import tensorflow as tf
from dataset.dataset import Dataset as Base
import numpy as np
from utility.proxy.realObject import RealObject


class FashionMNISTDataset(Base):

    def train_test_split(self):
        data = self.fetch_dataset()

        (train_images, train_labels), (test_images, test_labels) = data.load_data()

        train_images = np.pad(
            train_images, ((0, 0), (2, 2), (2, 2)), 'constant')
        test_images = np.pad(test_images, ((0, 0), (2, 2), (2, 2)), 'constant')

        print(train_images.shape)
        print(train_labels.shape)
        print(test_images.shape)
        print(test_labels.shape)

        return train_images, train_labels, test_images, test_labels

    def fetch_dataset(self):
        data = tf.keras.datasets.fashion_mnist
        return data


    def draw_sample(self):
        pass


class FashionMNIST(RealObject):

    def __init__(self, adaptee):
        if isinstance(adaptee, FashionMNISTDataset) :
            super(FashionMNIST, self).__init__(adaptee)
        else:
            raise Exception("Adapting fashion mnist dataset failed")

    def request(self, params):
        print("Load dataset ...")
        return self._adaptee.train_test_split()