from models.models import SimilarityModel
from models.factory import FeatureExtractModelFactory,  SiameseModelFactory
from loader.loader import Loader
import numpy as np
import pytest
loader = Loader()

IMAGE_SIZE = 32

extractor_factory = FeatureExtractModelFactory((IMAGE_SIZE, IMAGE_SIZE, 1))

simi_factory = SiameseModelFactory(
    extractor_factory
)

def test_model_similarity_preprocess():
    model = SimilarityModel(simi_factory)
    train_groups, test_group = model.preprocess(loader.load("fashion_mnist"))
    assert np.shape(train_groups) == (10, 6000, 32, 32, 1)
    assert np.shape(test_group) == (10, 1000, 32, 32, 1)

def test_model_similarity_all():
    model = SimilarityModel(simi_factory)
    train_groups, test_group = model.preprocess(loader.load("fashion_mnist"))
    model.compile()
    model.train(train_groups, test_group,  epochs = 5, steps_per_epoch = 10, batch_size = 32)
    model.evaluate()
    model.plotHistory()

def test_model_save_restore():
    pass
