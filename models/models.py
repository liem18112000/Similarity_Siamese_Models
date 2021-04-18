from models.factory import *
import matplotlib.pyplot as plt
from models.utility import *
from datetime import datetime

class AbstractModel(object):

    def __init__(self):
        self._factory = None
        self._model = None
        self._generator = None
        self._history = None

    def preprocess(self, dataset):
        try:
            x_train, y_train, x_test, y_test = dataset
            train_groups, test_groups = preprocess_data_into_groups(train=(x_train, y_train), test=(x_test, y_test), image_size=32)
        except:
            raise Exception("Preprocess dataset failed")
        else:
            print("Preprocess dataset ...")
        
        return train_groups, test_groups

    def compile(self):
        if self._factory is None:
            raise Exception("Model factory is not initialized")
        else:
            print("Model compiling ....")

    def train(self, train_groups, val_groups, epochs=20, steps_per_epoch=100, batch_size=1024):
        if self._model is None:
            raise Exception("Model is not initialized amd compiled")
        else:
            print("Model training ....")

    def evaluate(self, test_groups):
        if self._model is None:
            raise Exception("Model is not initialized amd compiled")
        else:
            print("Model evaluating ....")

    def predict(self, X):
        if self._model is None:
            raise Exception("Model is not initialized amd compiled")
        else:
            return self._model.predict(X)

    def plotHistory(self):
        if self._history is None:
            raise Exception("Model has not trained yet")
        else:
            print("Model analysis ....")

    def save(self, filename):
        if self._model is None:
            raise Exception("Model is not initialized amd compiled")
        else:
            if filename is None:
                filename = "trained_models/model_" + str(datetime.now())
            else:
                filename = "trained_models/" + filename
            self._model.save(filename)
            print("Save model : '" + filename + "'")

    def load(self, filename):
        return tf.keras.models.load_model('trained_models/' + str(filename))

    def summary(self):
        if self._model is None:
            raise Exception("Model is not initialized amd compiled")
        else:
            self._model.summary()


class SimilarityModel(AbstractModel):

    def __init__(self, factory):
        super(SimilarityModel, self).__init__()
        self._factory = factory

    def compile(self):

        super(SimilarityModel, self).compile()
        self._model = self._factory.createModel()
        self._generator = self._factory.createDataGenerator()
        _loss = self._factory.createLossFunction()

        # setup the optimization process
        self._model.compile(
            optimizer='adam',
            loss=_loss,
            metrics=['mae', 'acc']
        )

    def train(self, train_groups, val_groups, epochs = 20, steps_per_epoch = 100, batch_size = 1024):
        super(SimilarityModel, self).train(train_groups, val_groups, epochs, steps_per_epoch)
        self._history = self._model.fit(
            self._generator(train_groups, batch_size),
            steps_per_epoch=steps_per_epoch,
            validation_data=(self._generator(val_groups, batch_size)),
            epochs=epochs,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor='loss', patience=int(np.ceil(np.sqrt(epochs))), restore_best_weights=True)]
        )
    
    def evaluate(self, test_groups):
        super(SimilarityModel, self).evaluate()
        return self._model.evaluate(self._generator(test_groups), batch_size=256, steps=100)

    def plotHistory(self):
        super(SimilarityModel, self).plotHistory()
        history = self._history

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


class SiameseModel(AbstractModel):

    def __init__(self, factory):
        super(SiameseModel, self).__init__()
        self._factory = factory

    def compile(self):

        super(SiameseModel, self).compile()
        self._model = self._factory.createModel()
        self._generator = self._factory.createDataGenerator()
        _loss = self._factory.createLossFunction()

        # setup the optimization process
        self._model.compile(
            optimizer='adam',
            loss=_loss
        )

    def train(self, train_groups, val_groups = None, epochs=20, steps_per_epoch=100, batch_size=1024, image_size = 32):
        super(SiameseModel, self).train(train_groups, val_groups, epochs, steps_per_epoch)
        x_train, y_train = train_groups

        # we want a constant validation group to have a frame of reference for model performance
        self._history = self._model.fit(
            _generator(x_train, y_train, image_size=image_size, emb_size=512),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor='loss', patience=4, restore_best_weights=True)]
        )

    def evaluate(self, test_groups):
        super(SimilarityModel, self).evaluate()
        x_test, y_test = test_groups
        return self._model.evaluate(
            self._generator(x_test, y_test, image_size=32, emb_size=512),
            steps=100,
        )

    def plotHistory(self):
        super(SiameseModel, self).plotHistory()
        history = self._history

        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
