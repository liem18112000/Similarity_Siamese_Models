from models.factory import FeatureExtractModelFactory,  SiameseModelFactory
from loader.loader import Loader
import numpy as np
from models.utility import preprocess_data_into_groups
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import date, datetime
IMAGE_SIZE = 32
CHANNEL = 3
BATCH_SIZE = 256

def prepare_data():

    loader = Loader()

    train_images, train_labels, test_images, test_labels = loader.load("bollywood")

    train_groups, test_groups = preprocess_data_into_groups(
        train=(train_images, train_labels), test=(test_images, test_labels), image_size=IMAGE_SIZE, channel=CHANNEL)

    return train_groups, test_groups



def create_model():

    extractor_factory = FeatureExtractModelFactory((IMAGE_SIZE, IMAGE_SIZE, CHANNEL))

    factory = SiameseModelFactory(
        extractor_factory
    )

    _model = factory.createModel()

    _generator = factory.createDataGenerator()

    _loss = factory.createLossFunction()

    return _model, _generator, _loss



def plot(history):

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

import os
def save(model, filename):
    today = date.today()
    path = "trained_models"
    if not os.path.exists(path):
        os.makedirs(path)
    filename = path + '/' + str(filename) + "_" + str(datetime.now())
    print("Model saved at : " + filename)
    model.save(filename)

def main():

    train_groups, test_groups = prepare_data()

    _model, _generator, _loss= create_model()

    # setup the optimization process
    _model.compile(
        optimizer='adam',
        loss=_loss,
        metrics=['mae', 'acc']
    )

    # we want a constant validation group to have a frame of reference for model performance
    history = _model.fit(
        _generator(train_groups, BATCH_SIZE),
        steps_per_epoch=10,
        validation_data=(_generator(test_groups, BATCH_SIZE)),
        validation_steps=1,
        epochs=10,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=4, restore_best_weights=True)]
    )

    plot(history)

    save(_model, 'similarity_bollywood')


if __name__ == "__main__":
    main()
