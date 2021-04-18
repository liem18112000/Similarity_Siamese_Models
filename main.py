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
    train_groups, test_groups = model.preprocess(loader.load("fashion_mnist"))
    model.compile()
    model.train(train_groups, test_groups)
    model.evaluate(test_groups)
    model.plotHistory()


if __name__ == "__main__":
    main()
