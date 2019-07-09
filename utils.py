# Imports
import numpy as np
from keras import backend as K
from keras import optimizers
from keras.datasets import mnist, cifar10
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, BatchNormalization, Input, Dropout, Concatenate, Flatten
from keras.utils import to_categorical, Sequence, plot_model
from keras.layers.merge import add
from keras.callbacks import Callback, ModelCheckpoint
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from verbose import Verbose

"""
Functions
"""


def reset_layers(model, layers):
    assert max(layers) < len(model.layers)
    session = K.get_session()
    for idx in layers:
        layer = model.layers[idx]
        for v in layer.__dict__:
            v_arg = getattr(layer, v)
            if hasattr(v_arg, 'initializer'):
                initializer_method = getattr(v_arg, 'initializer')
                initializer_method.run(session=session)


def calc_robustness(test_data, model, source_weights_model=None, layer_indices=[], batch_size=32):
    """
    Evaluates the model on test data.

    Optionally, if source_weights_model is given, sets the weights
    of the model (for layer indices appearing in layer_indices) at
    each given layer to the weights of source_weights_model.

    :param test_data: A tuple (x_test, y_test) for validation
    :param model: The model to evaluate
    :param source_weights_model: The model from which we should
        copy weights and update our model's weights before eval.
    :param layer_indices: Layers to reset the weights of.
    :param batch_size: Self explanatory
    :return: A number in the interval [0,1] representing accuracy.
    """
    x_test, y_test = test_data
    if source_weights_model:
        for idx in layer_indices:
            loaded_weights = source_weights_model.layers[idx].get_weights()
            model.layers[idx].set_weights(loaded_weights)
    evaluated_metrics = model.evaluate(x_test, y_test, batch_size=batch_size)
    return evaluated_metrics[model.metrics_names.index('acc')]


def reset_to_checkpoint(model, checkpoint_weights):
    for idx in range(len(checkpoint_weights)):
        model.layers[idx].set_weights(checkpoint_weights[idx])
