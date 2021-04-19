import zipfile
from loader.loader import Loader
import numpy as np
import pytest

loader = Loader()

def test_loader_FashionMNIST():
    x_train, y_train, x_test, y_test = loader.load("fashion_mnist")
    assert np.shape(x_train) == (60000, 32, 32, )
    assert np.shape(y_train) == (60000, )
    assert np.shape(x_test) == (10000, 32, 32, )
    assert np.shape(y_test) == (10000, )


def test_loader_LWFPeople():
    params = {
        'test_rate' : 0.2,
        "random_state" : 42
    } 
    x_train, y_train, x_test, y_test = loader.load("lfwp", params)
    assert np.shape(x_train) == (1078, 32, 32, )
    assert np.shape(y_train) == (1078, )
    assert np.shape(x_test) == (270, 32, 32, )
    assert np.shape(y_test) == (270, )


def test_loader_Bollywood():
    x_train, y_train, x_test, y_test = loader.load("bollywood")

    assert np.shape(x_train) == (1078, 32, 32, )
    assert np.shape(y_train) == (1078, )
    assert np.shape(x_test) == (270, 32, 32, )
    assert np.shape(y_test) == (270, )
    # assert np.shape(x_train) == (1078, 32, 32, )
    # assert np.shape(y_train) == (1078, )
    # assert np.shape(x_test) == (270, 32, 32, )
    # assert np.shape(y_test) == (270, )



