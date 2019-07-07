# Imports
from keras import backend as K
from keras import optimizers
import numpy as np
from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers import Activation, Dense, BatchNormalization, Input, Dropout, Concatenate, Flatten
from keras.utils import to_categorical, Sequence, plot_model
from keras.models import Model, Sequential
from keras.layers.merge import add
from keras.callbacks import Callback, ModelCheckpoint
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16

# Drive mounting
# from google.colab import drive
# drive.mount('/content/drive')
# Define the working directory everyone should use
ROOT_DIR = '/content/drive/My Drive/globi/'

"""
Functions
"""


def copy_layers(target, source, layers):
    assert len(target.layers) == len(source.layers)
    assert max(layers) < len(target.layers)
    for idx in layers:
        target.layers[idx].set_weights(source.layers[idx].get_weights())


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


def get_dataset_name(dataset):
    name = dataset.__name__
    return name[name.rfind(".") + 1:]


def get_epoch_checkpoints():
    return [0, 1, 2, 3, 8, 40, 90, 100]


def save_model(model, filepath, weights_only=False):
    if weights_only:
        model.save_weights(filepath, overwrite=True)
    else:
        model.save(filepath, overwrite=True)
# TODO: Copy the load_model function


def get_layers_weights(model):
# TODO: Make this one line
    weights = []
    for layer in model.layers:
        weights.append(layer.get_weights())
    return weights


def calc_robustness(test_data, model, epochs_weights, layer_indices=[], batch_size=32):
    results = {}
    x_test, y_test = test_data
    results["baseline"] = model.evaluate(x_test, y_test, batch_size=batch_size)
    for epoch, weights in epochs_weights.items():
        for idx in layer_indices:
            model.layers[idx].set_weights(weights[idx])
        results[epoch] = model.evaluate(x_test, y_test, batch_size=batch_size)
    return results


def reset_to_checkpoint(model, checkpoint_weights):
    for idx in range(len(checkpoint_weights)):
        model.layers[idx].set_weights(checkpoint_weights[idx])


"""
Callbacks
"""


class SaveEpochsWeightsToDictCheckpoint(Callback):
    def __init__(self, weights_dict, period=[]):
        super(SaveEpochsWeightsToDictCheckpoint, self).__init__()
        self.weights_dict = weights_dict
        self.period = period

    def on_train_begin(self, logs=None):
        self.weights_dict["train_begin"] = get_layers_weights(self.model)

    def on_epoch_end(self, epoch, logs=None):
        if epoch in self.period:
            self.weights_dict[str(epoch)] = get_layers_weights(self.model)

    def on_train_end(self, logs=None):
        self.weights_dict["train_end"] = get_layers_weights(self.model)


class CustomModelCheckpoint(Callback):
    def __init__(self, filepath, save_weights_only=False, period=None):
        super(CustomModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.save_weights_only = save_weights_only
        self.period = period if period else []

    def on_train_begin(self, logs=None):
        filepath = self.filepath.format(epoch="start", **logs)
        save_model(self.model, filepath, self.save_weights_only)
        print("saved model to {}".format(filepath))

    def on_epoch_end(self, epoch, logs=None):
        if epoch in self.period:
            filepath = self.filepath.format(epoch=epoch, **logs)
            save_model(self.model, filepath, self.save_weights_only)
            print("saved model to {}".format(filepath))

    def on_train_end(self, logs=None):
        filepath = self.filepath.format(epoch="end", **logs)
        save_model(self.model, filepath, self.save_weights_only)
        print("saved model to {}".format(filepath))


"""
Classes
"""


class FrozenDenseFit:
    def __init__(self,
                 dataset,
                 verbose=False,
                 n_layers=3,
                 n_classes=10,
                 n_neurons=256,
                 epochs=100,
                 batch_size=32,
                 activation='relu',
                 output_activation='softmax',
                 layers_to_freeze=None,
                 weight_map=None,
                 optimizer=optimizers.SGD(momentum=0.9, nesterov=True),
                 loss='sparse_categorical_crossentropy',
                 metrics=None):
        self._dataset = dataset
        self._verbose = verbose
        self._n_layers = n_layers
        self._n_classes = n_classes
        self._n_neurons = n_neurons
        self._epochs = epochs
        self._batch_size = batch_size
        self._activation = activation
        self._output_activation = output_activation
        self._layers_to_freeze = layers_to_freeze if layers_to_freeze else []
        self._weight_map = weight_map if weight_map else {}
        self._optimizer = optimizer
        self._loss = loss
        self._metrics = metrics if metrics else ['accuracy']
        # Load the data at this point to set the shape
        (self._x_train, self._y_train), (self._x_test, self._y_test) = self._load_data_normalized()
        self._shape = (np.prod(self._x_train.shape[1:]),)
        self._print("Data shape: {}".format(self._shape))

    def _print(self, *args, **kwargs):
        if self._verbose:
            print(*args, **kwargs)

    def _load_data_normalized(self):
        self._print("Loading and normalizing data.")
        (x_train, y_train), (x_test, y_test) = self._dataset.load_data()
        self._print("Before reshape:")
        self._print("x_train.shape: {}".format(x_train.shape))
        self._print("x_test.shape: {}".format(x_test.shape))
        self._print("y_train.shape: {}".format(y_train.shape))
        self._print("y_test.shape: {}".format(y_test.shape))
        self._print("x_train[0][0][0] is {}".format(x_train[0][0][0]))
        self._print("x_train.astype('float32')[0][0][0] is {}".format(x_train.astype('float32')[0][0][0]))
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        self._print("x_train[0][0][0] is now {}".format(x_train[0][0][0]))
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
        self._print("After reshape:")
        self._print("x_train.shape: {}".format(x_train.shape))
        self._print("x_test.shape: {}".format(x_test.shape))
        return (x_train, y_train), (x_test, y_test)

    def _create_layers_with_freeze(self):
        self._print("Creating {} layers, freezing {}".format(self._n_layers, len(self._layers_to_freeze)))
        # Input layer gets special treatment:
        self._print("Input layer initialized with shape={}".format(self._shape))
        input_layer = Input(shape=self._shape)
        if 0 in self._weight_map:
            input_layer.set_weights(self._weight_map[0])
        layers = [input_layer]
        # Now the rest of the scum:
        for i in range(self._n_layers):
            if i == self._n_layers - 1:  # Output layer also gets special treatment
                layer = Dense(self._n_classes, activation=self._output_activation)
            else:
                layer = Dense(self._n_neurons, activation=self._activation)
            if i in self._layers_to_freeze:
                layer.trainable = False
            if i in self._weight_map:
                layer.set_weights(self._weight_map[i])
            layers += [layer]
        self._print("Done, returning layer list of length {}".format(len(layers)))
        return layers

    def _connect_layers(self, layers):
        # Connect first layer to input, then connect each subsequent
        # layer to the previous connected layer.
        self._print("Connecting {} layers...".format(len(layers)))
        input_layer = layers[0]
        next_layer = layers[1]  # Could be output
        connected_layers = [next_layer(input_layer)]
        for i in range(2, len(layers)):
            current_layer = layers[i]
            last_connected = connected_layers[-1]
            connected_layers += [current_layer(last_connected)]
        self._print("Done")
        return connected_layers

    def _get_epoch_checkpoint_callback(self):
        # For 100 epochs we'll get checkpoints at epochs 0, 1, 2, 3, 8, 40, 90, 100.
        # For N epochs we'll get checkpoints at 0, 0.01N, 0.02N, 0.03N, 0.08N, 0.4N,
        # 0.9N, N (rounded down)
        checkpoint_epochs = [(percentile * self._epochs) // 100 for percentile in get_epoch_checkpoints()]
        filepath = "vanilla.{epoch}.hdf5"
        return CustomModelCheckpoint(filepath, period=checkpoint_epochs)

    # Given a dataset, constructs a model with the requested parameters
    # and runs it after freezing layers. Also, optionally uses specified
    # weights.
    #
    # The dataset object should have a load_data() method (named in
    # keras.datasets).
    #
    # The shape parameter should be (28**2,) for mnist and (32**2,)
    # for cifur10.
    #
    # Including input and output layers, we'll have a total of n_layers+2
    # layers in the resulting topology.
    #
    # The layers_to_freeze parameter expects a list of layer indexes
    # in the range 0,1,2...n_layers+1 (yes, plus one, we're counting
    # input and output layers as well).
    #
    # The weight_map, if defined, should map layer indexes (in the
    # range [0,n_layers+1]) to weights (as output by layer.get_weights()).
    def go(self):
        # Load dataset, normalize and reshape
        layers = self._create_layers_with_freeze()
        connected_layers = self._connect_layers(layers)
        # TODO: Shouldn't layers[0] be connected_layers[0]?
        model = Model(layers[0], connected_layers[-1])
        model.compile(optimizer=self._optimizer, loss=self._loss, metrics=self._metrics)
        self._print(model.summary())
        model.fit(self._x_train, self._y_train,
                  shuffle=True,
                  epochs=self._epochs,
                  callbacks=[self._get_epoch_checkpoint_callback()],
                  batch_size=self._batch_size,
                  validation_data=(self._x_test, self._y_test))
        return model
