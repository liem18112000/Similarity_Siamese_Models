import tensorflow as tf
from dataset.dataset import Dataset as Base
import numpy as np
from utility.proxy.realObject import RealObject
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import zipfile
import shutil


def mergefolders(root_src_dir, root_dst_dir):
    for src_dir, dirs, files in os.walk(root_src_dir):
        dst_dir = src_dir.replace(root_src_dir, root_dst_dir, 1)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for file_ in files:
            src_file = os.path.join(src_dir, file_)
            dst_file = os.path.join(dst_dir, file_)
            if os.path.exists(dst_file):
                os.remove(dst_file)
            shutil.copy(src_file, dst_dir)
    print('Merge Completed Successfully!!')


class BollywoodDataset(Base):
    def draw_sample(self):
        pass

    def fetch_dataset(self):

        if not os.path.exists('temp/bollywood/train'):
            os.makedirs('temp/bollywood/train')

        if not os.path.isfile('temp/bollywood/100-bollywood-celebrity-faces.zip'):
            os.system("cd temp/bollywood && kaggle datasets download -d havingfun/100-bollywood-celebrity-faces ")
            with zipfile.ZipFile("temp/bollywood/100-bollywood-celebrity-faces.zip", 'r') as zip_ref:
                zip_ref.extractall("temp/bollywood")

        train_dirs = os.listdir('temp/bollywood')
        train_dirs.remove("100-bollywood-celebrity-faces.zip")
        merged_path = 'temp/bollywood/train/'

        if len(os.listdir(merged_path)) <= 0:
            for dir in train_dirs:
                mergefolders("temp/bollywood/" + str(dir), merged_path)

        return merged_path


    def train_test_split(self):
        train_path = self.fetch_dataset()
        datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True
        )
        x_train, y_train, x_test, y_test = [], [], [], []
        generator = datagen.flow_from_directory(
            train_path, batch_size=1024, target_size=(32, 32), class_mode="sparse")

        x, y = generator.next()
        x_train.append(x.copy())
        y_train.append(y.copy())

        x, y = generator.next()
        x_test.append(x.copy())
        y_test.append(y.copy())

        return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

class Bollywood(RealObject):

    def __init__(self, adaptee):
        if isinstance(adaptee, BollywoodDataset):
            super(Bollywood, self).__init__(adaptee)
        else:
            raise Exception("Adapting bollywood dataset failed")

    def request(self, params):
        print("Load dataset ...")
        return self._adaptee.train_test_split()
