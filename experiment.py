from datetime import datetime
import matplotlib.pylab as plt
import os
import pytz
import seaborn as sns
from kardashigans.resource_manager import ResourceManager
from kardashigans.verbose import Verbose
from keras import backend as K


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
        self._trained_models = {}
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
    def calc_robustness(test_data, model, source_weights_model=None, layer_indices=[], batch_size=32):
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

    def _dataset_fit(self, model_name, force=False):
        """
        Trains a model, or loads from file if a path exists using the
        ResourceManager.

        If the model was already trained locally, simply returns the existing
        model (unless forced to re-train).

        :param model_name: The (unique) name of the model to train / load.
        :param force: If set to True, will re-train the model even if it was
            locally trained. Also overrides loading model from disk.
        :return: The trained model.
        """
        assert model_name in self._model_names, \
            "Model '{}' not listed in {}".format(model_name, self._model_names)
        if force or (model_name not in self._trained_models):
            if not force:
                self.try_load_model(model_name)
            # If the load failed, or for some reason we need to fit the model:
            if model_name not in self._trained_models:
                self._print("Fitting dataset {}".format(model_name))
                model = self._trainers[model_name].go()
                self._trained_models[model_name] = model
                self._resource_manager.save_model(model, model_name)
                self._post_fit(model_name)
        else:
            self._print("Already trained model for {}, returning it".format(model_name))
        return self._trained_models[model_name]

    def _post_fit(self, model_name):
        pass

    def try_load_model(self, model_name):
        """ Tries to read disk and load model to _trained_models """
        model = self._resource_manager.try_load_model(model_name)
        if model:
            self._trained_models[model_name] = model

    def go(self):
        """ Implement this in inheriting classes """
        pass


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

    def _try_load_model_with_checkpoints(self, model_name):
        model = self._resource_manager.try_load_model(model_name)
        if not model:
            return None
        model.saved_checkpoints = self._resource_manager.try_load_model_checkpoints(
            model_name=model_name,
            period=self.get_epoch_save_period()
        )
        return model

    def try_load_model(self, model_name):
        """ Override this (from superclass) to load checkpoints also """
        self._print("In try_load_model, ExperimentWithCheckpoints version")
        model = self._try_load_model_with_checkpoints(model_name)
        if model:
            self._trained_models[model_name] = model

    def _post_fit(self, model_name):
        model_with_checkpoints = self._try_load_model_with_checkpoints(model_name)
        if model_with_checkpoints:
            self._trained_models[model_name].saved_checkpoints = model_with_checkpoints.saved_checkpoints
        else:
            self._print("Couldn't load checkpoint data for model {}".format(model_name))
