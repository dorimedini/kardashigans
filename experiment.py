from datetime import datetime
import os
import pytz
from kardashigans.resource_manager import ResourceManager
from kardashigans.verbose import Verbose
from kardashigans.analyze_model import AnalyzeModel


class Experiment(Verbose):
    """ Base class for all experiments. """

    def __init__(self,
                 name,
                 model_names,
                 trainers,
                 root_dir,
                 resource_load_dir=None):
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
        """
        super(Experiment, self).__init__(name=name)
        self._name = name
        self._model_names = model_names
        self._trainers = trainers
        self._root_dir = root_dir
        self._resource_load_dir = resource_load_dir  # Gets default value if None
        self._setup_env()
        self._resource_manager = ResourceManager(save_dir=self._output_dir,
                                                 load_dir=self._resource_load_dir)
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
        self._output_dir = self._base_dir + self._time_started + "/"
        if not self._resource_load_dir:
            self._resource_load_dir = self._output_dir
        else:
            self._resource_load_dir = self._root_dir + self._resource_load_dir
        self.logger.debug("In _setup_env(), setting up test dir at {}".format(self._output_dir))
        if not os.path.isdir(self._base_dir):
            os.mkdir(self._base_dir)
        if not os.path.isdir(self._output_dir):
            os.mkdir(self._output_dir)
        self.logger.debug("Test dir setup complete")

    def _init_test_data(self):
        self._test_sets = {}
        for model_name in self._model_names:
            self._test_sets[model_name] = {}
            x_test, y_test = self._trainers[model_name].get_test_data()
            self._test_sets[model_name]['x'] = x_test
            self._test_sets[model_name]['y'] = y_test

    def _save_results(self, results, model_name, results_name):
        self._resource_manager.save_results(results, model_name, results_name)

    def _load_results(self, model_name, results_name):
        return self._resource_manager.load_results(model_name, results_name)

    def _update_results(self, results, model_name, results_name):
        self._resource_manager.update_results(results, model_name, results_name)

    def _save_model(self, model, name):
        self._resource_manager.save_model(model=model, model_name=name)

    def _load_model(self, model_name):
        return self._resource_manager.load_model(model_name)

    def get_test_data(self, model_name):
        test_set = self._test_sets[model_name]
        return test_set['x'], test_set['y']

    @staticmethod
    def get_dataset_name(dataset):
        name = dataset.__name__
        return name[name.rfind(".") + 1:]

    def get_trainer_map(self):
        return self._trainers

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
            self.logger.warning("Failed to load model {} from disk".format(model_name))
            self.logger.debug("Fitting dataset {}".format(model_name))
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
    def __init__(self, *args, period, use_prev_results=True, **kwargs):
        """
        :param period: A list of epoch numbers at which callbacks should
            be called.
        :param use_prev_results: If True, before evaluating a model the experiment
            will first attempt to load the results dictionary from the filesystem.
            Any key in the loaded object will be skipped in get_full_robustness_results.
        """
        super(ExperimentWithCheckpoints, self).__init__(*args, **kwargs)
        self._use_prev_results = use_prev_results
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

    def get_full_robustness_results(self, model_name: str, checkpoint_epochs: list, layer_indices_list: list,
                                    batch_size=32, relative_accuracy=False, save=True):
        """
        Calculates robustness results for all requested layer_indices for all epochs.

        By default, saves evaluation results to disk.

        :param model_name: model_name to load model and test_data
        :param checkpoint_epochs: the epochs that have checkpoint saved models
        :param layer_indices_list: a 2d list of layer indices to calculate robustness for.
        :param batch_size: batch_size for evaluation
        :param relative_accuracy: should the robustness result be absolute accuracy of model or relative to clean model
        :param save: If False, results aren't dumped to disk
        :return: 2d dictionary with acc values per layer indices per epoch
        """

        def return_acc(acc: float, clean_acc: float, relative=False):
            return acc / clean_acc if relative else acc

        test_data = self.get_test_data(model_name)
        clean_results_name = 'robustness_clean'
        dirty_results_name = 'robustness_dirty'
        prev_results = {}
        prev_clean = {}
        if self._use_prev_results:
            prev_results = self._resource_manager.get_existing_results(model_name, dirty_results_name)
            prev_clean = self._resource_manager.get_existing_results(model_name, clean_results_name)
            if prev_results:
                print("shouldn't get here")
                # Start by checking if there's anything to do.
                # If all requested results exist we should just return them.
                no_new_layer_sets = set([str(layers) for layers in layer_indices_list]).issubset(set(prev_results.keys()))
                no_new_epochs = set([str(e) for e in checkpoint_epochs]).issubset(set(list(prev_results.values())[0].keys()))
                self.logger.debug("Previous results {} all layer sets in {}"
                                  "".format("contain" if no_new_layer_sets else "doesn't have",
                                            [str(layers) for layers in layer_indices_list]))
                self.logger.debug("Previous results {} all checkpoints in {}"
                                  "".format("contain" if no_new_epochs else "doesn't have",
                                            [str(epoch) for epoch in checkpoint_epochs]))
                if no_new_layer_sets and no_new_epochs:
                    self.logger.debug("Nothing new to compute, returning previous results")
                    return prev_results, prev_clean
        # Not all results are on disk (or we want to re-evaluate them)
        results = {str(layer_indices): {} for layer_indices in layer_indices_list}
        # clean
        with self.open_model(model_name) as model:
            if self._use_prev_results and prev_clean != {}:
                clean_eval = prev_clean
            else:
                clean_eval = AnalyzeModel.calc_robustness(
                    test_data=test_data,
                    model=model,
                    batch_size=batch_size)
            # rernd
            for layer_indices in layer_indices_list:
                if self._use_prev_results and str(layer_indices) in prev_results:
                    self.logger.debug("Using previously computed clean results for layer list {}"
                                      "".format(str(layer_indices)))
                    results[str(layer_indices)]["rernd"] = prev_results[str(layer_indices)]["rernd"]
                    continue
                curr_result = AnalyzeModel.calc_robustness(test_data=test_data,
                                                           model=model,
                                                           layer_indices=layer_indices,
                                                           batch_size=batch_size)
                results[str(layer_indices)]["rernd"] = return_acc(curr_result, clean_eval, relative_accuracy)
            # checkpoints
            for epoch in checkpoint_epochs:
                with self.open_model_at_epoch(model_name, epoch) as checkpoint_model:
                    try:
                        for layer_indices in layer_indices_list:
                            if self._use_prev_results and str(layer_indices) in prev_results:
                                self.logger.debug("Using previously computed epoch {} results for layer list {}"
                                                  "".format(str(epoch), str(layer_indices)))
                                results[str(layer_indices)][str(epoch)] = prev_results[str(layer_indices)][str(epoch)]
                                continue
                            curr_result = AnalyzeModel.calc_robustness(test_data=test_data,
                                                                       model=model,
                                                                       source_weights_model=checkpoint_model,
                                                                       layer_indices=layer_indices,
                                                                       batch_size=batch_size)
                            results[str(layer_indices)][str(epoch)] = return_acc(curr_result,
                                                                                 clean_eval, relative_accuracy)
                    except ValueError as e:
                        self.logger.error("Missing checkpoint at epoch {}, cannot continue".format(epoch))
                        self.logger.error("Exception: {}".format(e))
                        raise e
        if save:
            self._update_results(results, model_name, dirty_results_name)
            self._update_results(clean_eval, model_name, clean_results_name)
        return results, clean_eval
