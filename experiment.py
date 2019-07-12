from datetime import datetime
import matplotlib.pylab as plt
import os
import pytz
import seaborn as sns
from kardashigans.resource_manager import ResourceManager
from kardashigans.verbose import Verbose
from keras import backend as K
import numpy as np
import collections


class Experiment(Verbose):
    """ Base class for all experiments. """
    def __init__(self,
                 name,
                 model_names,
                 trainers,
                 root_dir,
                 resource_load_dir=None,
                 verbose=False):
        """
        Each experiment should have it's own unique name (it will get
        a folder named after it).

        Each participating model should be given a unique name so
        it can be saved and reloaded easily without re-training.

        :param name: Name of the specific experiment. This string is used as part
            of the file path when naming files output by the experiment / when loading
            files.
        :param model_names: A list of unique model names.
        :param trainers: Maps model names to Trainers. Each model should come with a
            Trainer object used to create it (no need to provide the models, only
            names and Trainers). If an existing path is found for the model, the
            Trainer is ignored.
        :param root_dir: path to the root directory from which models are loaded
            and to which models are saved. resource_load_dir, if provided, must be
            relative to the root_dir directory (if resource_load_dir provided,
            models loaded from root_dir + resource_load_dir)
        :param resource_load_dir: If provided, is the path to a directory from which
            models should be loaded, relative to the root directory. Provided as an
            argument to the ResourceManager.
        :param verbose: Logging on / off.
        """
        super(Experiment, self).__init__(verbose=verbose)
        self._name = name
        self._model_names = model_names
        self._trainers = trainers
        self._root_dir = root_dir
        self._resource_load_dir = resource_load_dir  # Gets default value if None
        self._setup_env()
        self._resource_manager = ResourceManager(model_save_dir=self._models_dir,
                                                 model_load_dir=self._resource_load_dir,
                                                 verbose=verbose)
        self._init_test_data()

    def _setup_env(self):
        """
        Creates the folder heirarchy for the experiment.

        At time of writing, it's:
            <ROOT_DRIVE_DIR>---<EXPERIMENT_NAME>---<TIME>-+-MODELS
                                                          |
                                                          +-RESULTS
        """
        self._base_dir = self._root_dir + self._name + "/"
        self._time_started = datetime.now(pytz.timezone('Israel')).strftime("%d_%m_%Y___%H_%M_%S")
        self._run_dir = self._base_dir + self._time_started + "/"
        self._results_dir = self._run_dir + "RESULTS/"
        self._models_dir = self._run_dir + "MODELS/"
        if not self._resource_load_dir:
            self._resource_load_dir = self._models_dir
        else:
            self._resource_load_dir = self._root_dir + self._resource_load_dir
        self._print("In _setup_env(), setting up test dir at {}".format(self._run_dir))
        if not os.path.isdir(self._base_dir):
            os.mkdir(self._base_dir)
        if not os.path.isdir(self._run_dir):
            os.mkdir(self._run_dir)
        if not os.path.isdir(self._results_dir):
            os.mkdir(self._results_dir)
        if not os.path.isdir(self._models_dir):
            os.mkdir(self._models_dir)
        self._print("Test dir setup complete")

    def _init_test_data(self):
        self._test_sets = {}
        for model_name in self._model_names:
            self._test_sets[model_name] = {}
            x_test, y_test = self._trainers[model_name].get_test_data()
            self._test_sets[model_name]['x'] = x_test
            self._test_sets[model_name]['y'] = y_test

    def _save_model(self, model, name):
        self._resource_manager.save_model(model=model, model_name=name)

    def _load_model(self, model_name):
        return self._resource_manager.load_model(model_name)

    def generate_heatmap(self, data, row_labels, col_labels, filename):
        """
        Creates a heatmap image from the data, outputs to file.

        :param data: List of lists of float values, indexed by data[row][column].
        :param row_labels: Size len(data) list of strings
        :param col_labels: Size len(data[0]) list of strings
        :param filename: Output filename, relative to the experiment results
            directory.
        """
        self._print("Generating heatmap. Data: {}".format(data))
        self._print("Rows: {}".format(row_labels))
        self._print("Cols: {}".format(col_labels))
        ax = sns.heatmap(data, linewidth=0.5, xticklabels=col_labels, yticklabels=row_labels)
        fig = ax.get_figure()
        fig.savefig(self._results_dir + filename)
        if self._verbose:
            plt.show()

    def get_test_data(self, model_name):
        test_set = self._test_sets[model_name]
        return test_set['x'], test_set['y']

    @staticmethod
    def get_dataset_name(dataset):
        name = dataset.__name__
        return name[name.rfind(".") + 1:]

    @staticmethod
    def rernd_layers(model, layers_indices):
        session = K.get_session()
        for idx in layers_indices:
            layer = model.layers[idx]
            for v in layer.__dict__:
                v_arg = getattr(layer, v)
                if hasattr(v_arg, 'initializer'):
                    initializer_method = getattr(v_arg, 'initializer')
                    initializer_method.run(session=session)

    @staticmethod
    def calc_robustness(test_data, model, source_weights_model=None, layer_indices=None, batch_size=32):
        """
        Evaluates the model on test data after re-initializing the layers
        with indices specified.

        Alternatively, if source_weights_model is given, sets the weights
        of the model (for layer indices appearing in layer_indices) at
        each given layer to the weights of source_weights_model.

        Function resets model weights to previous state before returning.

        :param test_data: A tuple (x_test, y_test) for validation
        :param model: The model to evaluate
        :param source_weights_model: The model from which we should
            copy weights and update our model's weights before eval.
        :param layer_indices: Layers to reset the weights of.
        :param batch_size: used in evaluation
        :return: A number in the interval [0,1] representing accuracy.
        """
        if not layer_indices:
            layer_indices = []
        x_test, y_test = test_data
        prev_weights = model.get_weights()
        if source_weights_model:
            for idx in layer_indices:
                loaded_weights = source_weights_model.layers[idx].get_weights()
                model.layers[idx].set_weights(loaded_weights)
        else:
            Experiment.rernd_layers(model, layer_indices)
        evaluated_metrics = model.evaluate(x_test, y_test, batch_size=batch_size)
        model.set_weights(prev_weights)
        return evaluated_metrics[model.metrics_names.index('acc')]

    @staticmethod
    def get_weight_distances(model, source_weights_model, layer_indices=[], norm_orders=[]):
        """
        Computes distances between the layers of the given model and source model, in the chosen layers.
        Returns a dictionary in format: {idx: [dists (in the same order as the given list of distances)]}.
        """
        distance_list = collections.defaultdict(list)
        for layer in layer_indices:
            source_weights = source_weights_model.layers[layer].get_weights()
            model_weights = model.layers[layer].get_weights()
            if source_weights and model_weights:
                source_flatten_weights = np.concatenate([source_w.flatten() for source_w in source_weights])
                model_flatten_weights = np.concatenate([model_w.flatten() for model_w in model_weights])
                for order in norm_orders:
                    distance_list[layer].append(
                        np.linalg.norm(model_flatten_weights - source_flatten_weights, ord=order))
        return distance_list

    def _get_model(self, model_name):
        """
        Used by the context manager. Tries to load the model, if it doesn't
        work the model is trained.
        """
        assert model_name in self._model_names, \
            "Model '{}' not listed in {}".format(model_name, self._model_names)
        try:
            return self._load_model(model_name)
        except:
            # If the load failed, or for some reason we need to fit the model:
            self._print("Fitting dataset {}".format(model_name))
            model = self._trainers[model_name].go()
            self._resource_manager.save_model(model, model_name)
            return model

    def open_model(self, model_name):
        """
        Use this to read trained model to local memory. Used in inheriting classes like:
        with self.open_model('mnist_fc3_vanilla') as model:
            model.get_weights()
            ...
        """
        return Experiment._model_context(self, model_name)

    def go(self):
        """ Implement this in inheriting classes """
        raise NotImplementedError

    class _model_context(object):
        def __init__(self, experiment, model_name):
            # The context needs access to Trainers / ResourceManager
            # so it makes sense to demand the calling experiment as
            # an interface.
            # No inheriting class will need to manually create a
            # _model_context anyway so the Experiment class can handle
            # the ugly
            self._exp = experiment
            self._model_name = model_name

        def __enter__(self):
            self._model = self._exp._get_model(self._model_name)
            return self._model

        def __exit__(self, exc_type, exc_val, exc_tb):
            del self._model


class ExperimentWithCheckpoints(Experiment):
    def __init__(self, *args, **kwargs):
        super(ExperimentWithCheckpoints, self).__init__(*args, **kwargs)
        self._add_epoch_checkpoint_callback()

    def get_epoch_save_period(self):
        return self._resource_manager.get_epoch_save_period()

    def _add_epoch_checkpoint_callback(self):
        for name in self._model_names:
            cb = self._resource_manager.get_epoch_save_callback(name)
            self._trainers[name].add_checkpoint_callback(cb)

    def _get_model_at_epoch(self, model_name, epoch):
        """
        This method may throw error! Unlike Experiment._get_model, we
        don't want to train the entire model just because we're missing
        an epoch, if you try to load an epoch from an untrained model
        you're going to have a bad time anyway.

        :param model_name: Model name
        :param epoch: A number, or 'start'/'end'
        :return: The loaded model
        """
        assert model_name in self._model_names, \
            "Model '{}' not listed in {}".format(model_name, self._model_names)
        model = self._resource_manager.try_load_model_at_epoch(model_name, epoch)
        if model:
            return model
        raise ValueError("Couldn't load model {} at epoch {}".format(model_name, epoch))

    def open_model_at_epoch(self, model_name, epoch):
        return ExperimentWithCheckpoints._model_at_epoch_context(experiment=self,
                                                                 model_name=model_name,
                                                                 epoch=epoch)

    class _model_at_epoch_context(Experiment._model_context):
        def __init__(self, experiment, model_name, epoch=None):
            Experiment._model_context.__init__(self, experiment=experiment, model_name=model_name)
            self._epoch = epoch

        def __enter__(self):
            if not self._epoch:
                self._model = self._exp._get_model(self._model_name)
            else:
                self._model = self._exp._get_model_at_epoch(self._model_name, self._epoch)
            return self._model
