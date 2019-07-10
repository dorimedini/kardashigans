# Save / load / train models.
from keras import optimizers
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
from kardashigans.verbose import Verbose


class FCTrainer(Verbose):
    """ Trains a fully connected network on a dataset """
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
                 optimizer=optimizers.SGD(momentum=0.9, nesterov=True),
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'],
                 checkpoint_callbacks=[]):
        """
        :param dataset: keras.datasets.mnist, for example
        :param verbose: Logging on / off
        :param n_layers: Number of layers in the FCN
        :param n_classes: Number of output classes
        :param n_neurons: Number of neurons per inner layer
        :param epochs: Number of epochs to train
        :param batch_size: Number of samples per batch
        :param activation: Activation function of inner nodes
        :param output_activation: Activation function of the output layer
        :param optimizer: Optimizer used when fitting
        :param loss: Loss used when fitting
        :param metrics: By which to score
        """
        super(FCTrainer, self).__init__(verbose=verbose)
        self._dataset = dataset
        self._n_layers = n_layers
        self._n_classes = n_classes
        self._n_neurons = n_neurons
        self._epochs = epochs
        self._batch_size = batch_size
        self._activation = activation
        self._output_activation = output_activation
        self._optimizer = optimizer
        self._loss = loss
        self._metrics = metrics
        self._checkpoint_callbacks = checkpoint_callbacks
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

    def _create_layers(self):
        self._print("Creating {} layers".format(self._n_layers))
        # Input layer gets special treatment:
        self._print("Input layer initialized with shape={}".format(self._shape))
        input_layer = Input(shape=self._shape)
        layers = [input_layer]
        # Now the rest of the scum:
        for i in range(self._n_layers):
            if i == self._n_layers - 1:  # Output layer also gets special treatment
                layer = Dense(self._n_classes, activation=self._output_activation)
            else:
                layer = Dense(self._n_neurons, activation=self._activation)
            layers += [layer]
        self._print("Done, returning layer list of length {}".format(len(layers)))
        return layers

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
        layers = self._create_layers()
        connected_layers = self._connect_layers(layers)
        # TODO: Shouldn't layers[0] be connected_layers[0]?
        model = Model(layers[0], connected_layers[-1])
        model.compile(optimizer=self._optimizer, loss=self._loss, metrics=self._metrics)
        self._print(model.summary())
        model.fit(self._x_train, self._y_train,
                  shuffle=True,
                  epochs=self._epochs,
                  callbacks=self._checkpoint_callbacks,
                  batch_size=self._batch_size,
                  validation_data=(self._x_test, self._y_test))
        return model

    def get_test_data(self):
        return self._x_test, self._y_test

    def set_checkpoint_callbacks(self, callbacks=[]):
        self._checkpoint_callbacks = callbacks


class FCFreezeTrainer(FCTrainer):
    """ Trainer used when we need to freeze layers.

        Overrides the _create_layers method of FCTrainer to allow layer
        freezing and weight initialization.
    """
    def __init__(self, layers_to_freeze=[], weight_map={}, **kwargs):
        """
        :param layers_to_freeze: Optional list of layer indexes to set to
            'untrainable', i.e. their weights cannot change during
            training.
        :param weight_map: Optional. Maps layer indexes to initial weight
            values to use (instead of random init). Intended for use
            with frozen layers.
        """
        super(FCFreezeTrainer, self).__init__(**kwargs)
        self._layers_to_freeze = layers_to_freeze
        self._weight_map = weight_map

    def _create_layers(self):
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
