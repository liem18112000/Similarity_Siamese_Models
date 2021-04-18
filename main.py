from models.models import SimilarityModel
from models.factory import FeatureExtractModelFactory,  SiameseModelFactory
from loader.loader import Loader
import numpy as np
import pytest

IMAGE_SIZE = 32


def main():
    loader = Loader()
    extractor_factory = FeatureExtractModelFactory((IMAGE_SIZE, IMAGE_SIZE, 1))
    simi_factory = SiameseModelFactory(
        extractor_factory
    )

    model = SimilarityModel(simi_factory)
    train_groups, test_group = model.preprocess(loader.load("fashion_mnist"))
    model.compile()
    model.train(train_groups, test_group,  epochs=10,steps_per_epoch=10, batch_size=32)
    model.evaluate(test_group, steps=1, batch_size=32)
    model.plotHistory()


if __name__ == "__main__":
    main()
