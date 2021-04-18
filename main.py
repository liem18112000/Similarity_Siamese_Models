from models.factory import FeatureExtractModelFactory,  SiameseModelFactory
from loader.loader import Loader
import numpy as np
from models.utility import preprocess_data_into_groups
import tensorflow as tf
import matplotlib.pyplot as plt

IMAGE_SIZE = 32

def prepare_data():

    loader = Loader()

    train_images, train_labels, test_images, test_labels = loader.load("fashion_mnist")

    train_groups, test_groups = preprocess_data_into_groups(train=(train_images, train_labels), test=(test_images, test_labels), image_size=IMAGE_SIZE)

    return train_groups, test_groups



def create_model():

    extractor_factory = FeatureExtractModelFactory((IMAGE_SIZE, IMAGE_SIZE, 1))

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


def main():

    train_groups, test_groups = prepare_data()

    _model, _generator, _loss= create_model()

    # setup the optimization process
    _model.compile(
        optimizer='adam',
        loss=_loss,
        metrics=['mae', 'acc']
    )

    inp, target = _generator(test_groups, 1024)

    # we want a constant validation group to have a frame of reference for model performance
    history = _model.fit(
        _generator(train_groups),
        steps_per_epoch=100,
        validation_data=(inp, target),
        epochs=20,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=4, restore_best_weights=True)]
    )

    plot(history)


if __name__ == "__main__":
    main()
