from datetime import datetime
import os
import pytz
from kardashigans.resource_manager import ResourceManager
from kardashigans.verbose import Verbose

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

    def _save_model(self, model, name):
        self._resource_manager.save_model(model=model, model_name=name)

    def _load_model(self, model_name):
        return self._resource_manager.load_model(model_name)

    @staticmethod
    def get_dataset_name(dataset):
        name = dataset.__name__
        return name[name.rfind(".") + 1:]

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
    def __init__(self, *args, period, **kwargs):
        """
        :param period: A list of epoch numbers at which callbacks should
            be called.
        """
        super(ExperimentWithCheckpoints, self).__init__(*args, **kwargs)
        self._add_epoch_checkpoint_callback(period)

    def _add_epoch_checkpoint_callback(self, period):
        for name in self._model_names:
            cb = self._resource_manager.get_epoch_save_callback(name, period)
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
        def __init__(self, experiment, model_name, epoch):
            Experiment._model_context.__init__(self, experiment=experiment, model_name=model_name)
            self._epoch = epoch

        def __enter__(self):
            self._model = self._exp._get_model_at_epoch(self._model_name, self._epoch)
            return self._model
