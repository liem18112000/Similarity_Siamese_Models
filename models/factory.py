import tensorflow as tf
from copy import deepcopy
from datetime import datetime
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from models.utility import *


class FeatureExtractModelFactory(object):

    def __init__(self, input_shape):
        self._input_shape_ = input_shape

    def createInputLayers(self, input_layer=None):
        if input_layer is None:
            input_layer = Input(shape=self._input_shape_)

        return input_layer

    def createConvBlocks(self, input_layer, num_conv_layer, num_sub_conv, kernel):
        n_layer = input_layer
        for i in range(num_conv_layer):

            new_kernel = kernel*2**i
            for j in range(num_sub_conv):
                n_layer = Conv2D(new_kernel*2**j, kernel_size=(3, 3),activation='linear', padding='same')(n_layer)
                n_layer = BatchNormalization()(n_layer)
                n_layer = Activation('relu')(n_layer)

            n_layer = MaxPool2D((2, 2))(n_layer)

        return n_layer

    def createDenseBlocks(self, input_layer, num_neural, num_block, use_dropout, dropout_rate):
        n_layer = input_layer
        n_layer = Flatten()(n_layer)

        for i in range(num_block):
            n_layer = Dense(num_neural, activation='linear')(n_layer)

            if use_dropout:
                n_layer = Dropout(dropout_rate)(n_layer)

            n_layer = BatchNormalization()(n_layer)

        n_layer = Activation('relu')(n_layer)
        return n_layer

    def createModel(self, name, input_params, conv_params, dense_params, show_summary=True):
        img_in = self.createInputLayers(
            input_params
        )

        n_layer = self.createConvBlocks(
            img_in,
            conv_params['num_conv_layer'],
            conv_params['num_sub_conv'] if 'num_sub_conv' in conv_params.keys(
            ) else 2,
            conv_params['kernel'] if 'kernel' in conv_params.keys() else 8
        )

        n_layer = self.createDenseBlocks(
            n_layer,
            dense_params['num_neural'] if 'num_neural' in dense_params.keys(
            ) else 32,
            dense_params['num_block'] if 'num_block' in dense_params.keys() else 1,
            dense_params['use_dropout'] if 'use_dropout' in dense_params.keys(
            ) else True,
            dense_params['dropout_rate'] if 'dropout_rate' in dense_params.keys(
            ) else 0.5,
        )

        feature_model = Model(inputs=[img_in], outputs=[n_layer], name=name)

        if show_summary:
            feature_model.summary()

        return feature_model


class SiameseModelFactory(object):
    def __init__(self, feature_factory):
        self._input_shape_ = feature_factory._input_shape_
        self._feature_factory_ = feature_factory

    def createDenseBlocks(self, input_layer, num_neural, num_block, use_dropout, dropout_rate):
        n_layer = input_layer

        for i in range(num_block):
            n_layer = Dense(num_neural, activation='linear')(n_layer)

            if use_dropout:
                n_layer = Dropout(dropout_rate)(n_layer)

            n_layer = BatchNormalization()(n_layer)

        n_layer = Activation('relu')(n_layer)
        return n_layer

    def createFeatureExtractLayers(self, name, input_layers, show_summary=False):
        factory = self._feature_factory_
        feature_model = factory.createModel(
            name=name,
            input_params=input_layers,
            conv_params={
                "num_filter": 64,
                "num_conv_layer": estimate_conv_layers(32)
            },
            dense_params={
                'num_neural': 512
            },
            show_summary=show_summary
        )
        return feature_model

    def createModel(self, show_summary=True):
        img_a_in = Input(shape=self._input_shape_, name='ImageA_Input')
        img_b_in = Input(shape=self._input_shape_, name='ImageB_Input')
        img_a_feat = self.createFeatureExtractLayers(
            name='FeatureExtractModel_A', input_layers=img_a_in, show_summary=True)
        img_b_feat = self.createFeatureExtractLayers(
            name='FeatureExtractModel_B', input_layers=img_b_in)

        combined_features = concatenate(
            [img_a_feat(img_a_in), img_b_feat(img_b_in)], name='merge_features')
        combined_features = Dense(512, activation='linear')(combined_features)
        combined_features = BatchNormalization()(combined_features)
        combined_features = Activation('relu')(combined_features)
        combined_features = Dense(256, activation='linear')(combined_features)
        combined_features = BatchNormalization()(combined_features)
        combined_features = Activation('relu')(combined_features)
        combined_features = Dense(128, activation='linear')(combined_features)
        combined_features = BatchNormalization()(combined_features)
        combined_features = Activation('relu')(combined_features)
        combined_features = Dense(64, activation='linear')(combined_features)
        combined_features = BatchNormalization()(combined_features)
        combined_features = Activation('relu')(combined_features)
        combined_features = Dense(1, activation='sigmoid')(combined_features)
        model = Model(inputs=[img_a_in, img_b_in], outputs=[combined_features], name='Similarity_Model')

        if show_summary:
            model.summary()

        return model

    def createDataGenerator(self):
        return siam_gen

    def createLossFunction(self):
        return tf.keras.losses.BinaryCrossentropy()


class SiameseTripletLossModelFactory(SiameseModelFactory):

    def __init__(self, feature_factory):
        super(SiameseTripletLossModelFactory, self).__init__(feature_factory)

    def createModel(self, show_summary=True):
        img_anchor_in = Input(shape=self._input_shape_,
                              name='Image_Anchor_Input')
        img_positive_in = Input(shape=self._input_shape_,
                                name='Image_Positive_Input')
        img_negative_in = Input(shape=self._input_shape_,
                                name='Image_Negative_Input')
        img_anchor_feat = self.createFeatureExtractLayers(
            name='FeatureExtractModel_Anchor', input_layers=img_anchor_in)
        img_positive_feat = self.createFeatureExtractLayers(
            name='FeatureExtractModel_Positive', input_layers=img_positive_in)
        img_negative_feat = self.createFeatureExtractLayers(
            name='FeatureExtractModel_Negative', input_layers=img_negative_in)

        combined_features = concatenate([img_anchor_feat(img_anchor_in), img_positive_feat(
            img_positive_in), img_negative_feat(img_negative_in)], name='merge_features')
        model = Model(inputs=[img_anchor_in, img_positive_in, img_negative_in], outputs=[combined_features], name='Siamese_TripletLoss_Model')

        if show_summary:
            model.summary()

        return model

    def createDataGenerator(self):
        return data_generator

    def createLossFunction(self):
        return triplet_loss


def triplet_loss(y_true, y_pred, emb_size=512, alpha=0.0002):
    anchor, positive, negative = y_pred[:, :emb_size], y_pred[:,emb_size:2*emb_size], y_pred[:, 2*emb_size:]
    positive_dist = tf.reduce_mean(tf.square(anchor - positive), axis=1)
    negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=1)
    return tf.maximum(positive_dist - negative_dist + alpha, 0.)


class ModelManager(object):
    def save_model(self, model, file_name=None):
        if file_name is None:
            file_name = "model_" + str(datetime.now())
        model.save(file_name)
        print("Save model : '" + file_name + "'")


manager = ModelManager()
