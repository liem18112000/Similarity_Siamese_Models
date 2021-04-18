import seaborn as sns
from sklearn.model_selection import train_test_split
import cv2
from sklearn.datasets import fetch_lfw_people
from dataset.dataset import Dataset as Base
import numpy as np
import matplotlib.pyplot as plt
from utility.proxy.realObject import RealObject

class lfwPeopleDataset(Base):

    def __init__(self, min_faces_per_person = 60):
        self._min_faces_per_person = min_faces_per_person

    def draw_sample(self):
        sns.set(rc={'figure.figsize': (11.7, 8.27)})
        fig, ax = plt.subplots(3, 5)
        for i, axi in enumerate(ax.flat):
            axi.imshow(self.faces.images[i], cmap='bone')
            axi.set(xticks=[], yticks=[],
                    xlabel=self.faces.target_names[self.faces.target[i]])

    def resize_image(self, image_size):
        images = []
        for img in self.faces.data:
            images.append(
                cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
            )
        return images

    def fetch_dataset(self, image_size=32):
        self.faces = fetch_lfw_people(min_faces_per_person=self._min_faces_per_person)
        return np.reshape(self.resize_image(image_size), (-1, image_size, image_size, )), self.faces.target, self.faces.target_names

    def train_test_split(self, test_rate=0.1, random_state=42):
        X, y, labels = self.fetch_dataset()
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=test_rate, random_state=random_state)

        print(x_train.shape)
        print(y_train.shape)
        print(x_test.shape)
        print(y_test.shape)

        return x_train, y_train, x_test, y_test


class lfwPeople(RealObject):

    def __init__(self, adaptee):
        if isinstance(adaptee, lfwPeopleDataset):
            super(lfwPeople, self).__init__(adaptee)
        else:
            raise Exception("Adapting fashion LFWPeople dataset failed")


    def request(self, params):
        print("Loading dataset ...")
        return self._adaptee.train_test_split(
            test_rate = params['test_rate'], 
            random_state = params['random_state']
        )
