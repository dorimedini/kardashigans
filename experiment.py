from datetime import datetime
from resource_manager import ResourceManager
import matplotlib.pylab as plt
import os
import pytz
import seaborn as sns
import utils as U


class Experiment:
    """ Base class for all experiments. """
    def __init__(self,
                 name,
                 model_names,
                 trainers,
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
        :param resource_load_dir: If provided, is the path to a directory from which
            models should be loaded, relative to the root directory. Provided as an
            argument to the ResourceManager.
        :param verbose: Logging on / off.
        """
        self._name = name
        self._model_names = model_names
        self._verbose = verbose
        self._trainers = trainers
        self._trained_models = {}
        self._setup_env()
        self._resource_load_dir = resource_load_dir if resource_load_dir else self._models_dir
        self._resource_manager = ResourceManager(model_save_dir=self._models_dir,
                                                 model_load_dir=self._resource_load_dir,
                                                 verbose=verbose)
        self._init_test_data()

    def _print(self, *args, **kwargs):
        if self._verbose:
            print(*args, **kwargs)

    def _setup_env(self):
        """
        Creates the folder heirarchy for the experiment.

        At time of writing, it's:
            <ROOT_DRIVE_DIR>---<EXPERIMENT_NAME>---<TIME>-+-MODELS
                                                          |
                                                          +-RESULTS
        """
        self._base_dir = U.ROOT_DIR + self._name + "/"
        self._time_started = datetime.now(pytz.timezone('Israel')).strftime("%d_%m_%Y___%H_%M_%S")
        self._run_dir = self._base_dir + self._time_started + "/"
        self._results_dir = self._run_dir + "RESULTS/"
        self._models_dir = self._run_dir + "MODELS/"
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
