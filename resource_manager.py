import json
import keras
import os
from kardashigans.verbose import Verbose
from keras.callbacks import Callback


class ResourceManager(Verbose):
    """ Handles saving / loading trained models """
    def __init__(self, save_dir, load_dir):
        """
        :param save_dir: All saved files will be stored at this directory,
            under a subdirectory named by the date / time of initialization
            of the resource manager.
        :param load_dir: All loaded files are loaded from this directory.
            Loaded files (models / evaluation data etc.) must be directly
            contained in this directory.
        """
        super(ResourceManager, self).__init__()
        self._save_dir = self._add_slash(save_dir)
        self._load_dir = self._add_slash(load_dir)
        self._model_file_template = "{model_name}.h5"

    def _add_slash(self, path_to_dir):
        if path_to_dir[-1] not in ['/', '\\']:
            path_to_dir += "/"
        return path_to_dir

    def _get_model_save_fullpath(self, model_name):
        return self._save_dir + self._model_file_template.format(model_name=model_name)

    def _get_model_load_fullpath(self, model_name):
        return self._load_dir + self._model_file_template.format(model_name=model_name)

    def _get_checkpoint_model_name(self, model_name, epoch):
        return model_name + "_epoch_{}".format(epoch)

    def _get_checkpoint_file_template(self, model_name):
        return model_name + "_epoch_{epoch}.h5"

    @staticmethod
    def get_checkpoint_epoch_keys(period):
        return ['start'] + period + ['end']

    def get_epoch_save_callback(self, model_name, period):
        filepath_template = self._save_dir + self._get_checkpoint_file_template(model_name)
        return ResourceManager.SaveModelAtEpochsCallback(filepath_template=filepath_template,
                                                         period=period)

    def _get_results_save_fullpath(self, model_name, results_name):
        return "{}{}_{}.json".format(self._save_dir, model_name, results_name)

    def _get_results_load_fullpath(self, model_name, results_name):
        return "{}{}_{}.json".format(self._load_dir, model_name, results_name)

    def save_results(self, results, model_name, results_name):
        """
        Saves results object to JSON file.

        File name will be derived by model name and the name of the
        specific result. To prevent accidental overwrite of different
        results for the same model, the results name parameter has no
        default value.

        :param results: A JSONable object to output
        :param model_name: The model name of the model analyzed
        :param results_name: Unique (up to model name) name of results
            object
        """
        with open(self._get_results_save_fullpath(model_name, results_name), 'w') as file:
            file.write(json.dumps(results))

    def load_results(self, model_name, results_name):
        """ save_results^{-1} """
        with open(self._get_results_load_fullpath(model_name, results_name), 'r') as file:
            return json.loads(file.read())

    def get_existing_results(self, model_name, results_name):
        """
        Basically a soft wrapper for the load_results method
        """
        try:
            return self.load_results(model_name, results_name)
        except Exception as e:
            self.logger.warning("No previous results found, dumping new results. "
                                "Model/results names: {}/{}".format(model_name, results_name))
            self.logger.warning("Exception raised: {}".format(e))
        return {}

    def update_results(self, results, model_name, results_name):
        """
        Merges input results with previous results (if they exist) and
        outputs merged dict to disk (overwriting previous results).

        If conflicting result keys exist nothing is promised! Use with
        care
        """
        prev_results = self.get_existing_results(model_name, results_name)
        merged_results = {**results, **prev_results}
        self.save_results(merged_results, model_name, results_name)

    def save_model(self, model, model_name):
        model.save(self._get_model_save_fullpath(model_name),
                   overwrite=True)

    def save_history(self, history, model_name):
        history_file_name = model_name + '_history.json'
        history_path = os.path.join(self._save_dir, history_file_name)
        with open(history_path, 'w') as file:
            file.write(json.dumps(history))

    def load_model(self, model_name, fullpath=None):
        if not fullpath:
            fullpath = self._get_model_load_fullpath(model_name)
        try:
            return keras.models.load_model(fullpath)
        except IOError:
            fullpath = self._get_model_save_fullpath(model_name)
            return keras.models.load_model(fullpath)

    def try_load_model(self, model_name):
        """
        If we're unsure the model exists but we'd like to use it if it is,
        call this method (returns None if model couldn't be loaded)
        """
        try:
            return self.load_model(model_name)
        except:
            self.logger.warning("Couldn't load model from {}. Attempting to load from saved model "
                                "directory (maybe newly trained)"
                                "".format(self._get_model_load_fullpath(model_name)))
            try:
                model = self.load_model(model_name, fullpath=self._get_model_save_fullpath(model_name))
                self.logger.debug("Failed to load from load dir, but successfully loaded model {} "
                                  "from {}".format(model_name, self._get_model_save_fullpath(model_name)))
                return model
            except:
                self.logger.warning("Couldn't even load model from {}"
                                    "".format(self._get_model_save_fullpath(model_name)))
        return None

    def try_load_model_at_epoch(self, model_name, epoch):
        return self.try_load_model(self._get_checkpoint_model_name(model_name, epoch))

    def try_load_model_checkpoints(self, model_name, period):
        """
        Loads saved models at specific epochs. Intended for use with
        get_epoch_save_callback (used by Experiment objects via the
        Trainer's add_callback method).

        Will attempt to load start / end models and a model for each
        epoch given in the period parameter, will print error if
        something isn't found.

        :param model_name: Identifies the model to load. Should be the
            same string used in get_epoch_save_callback when they were
            first saved.
        :param period: List of epoch indexes (integers) to load from.
        :return: A mapping from epochs to loaded models. The keys of the
            map should be 'start', 'end' and all keys in the period
            parameter (if successful)
        """
        epoch_model_map = {}
        start_model = self.try_load_model_at_epoch(model_name, 'start')
        end_model = self.try_load_model_at_epoch(model_name, 'end')
        if start_model:
            epoch_model_map['start'] = start_model
        else:
            self.logger.warning("No start epoch found (tried {})".format(start_model))
        if end_model:
            epoch_model_map['end'] = end_model
        else:
            self.logger.warning("No end epoch found (tried {})".format(end_model))
        for epoch in period:
            epoch_model = self.try_load_model_at_epoch(model_name, epoch)
            if epoch_model:
                epoch_model_map[epoch] = epoch_model
            else:
                self.logger.warning("Saved epoch {} not found at {}".format(epoch, self._get_checkpoint_model_name(model_name, epoch)))
        return epoch_model_map

    class SaveModelAtEpochsCallback(Callback):
        def __init__(self, filepath_template, period=None):
            super(ResourceManager.SaveModelAtEpochsCallback, self).__init__()
            self.filepath_template = filepath_template
            self.period = period if period else []
            self.v = Verbose(name=self.__class__.__name__)

        def on_train_begin(self, logs=None):
            filepath = self.filepath_template.format(epoch="start", **logs)
            self.model.save(filepath, overwrite=True)
            self.v.logger.debug("saved model to {}".format(filepath))

        def on_epoch_end(self, epoch, logs=None):
            if epoch in self.period:
                filepath = self.filepath_template.format(epoch=epoch, **logs)
                self.model.save(filepath, overwrite=True)
                self.v.logger.debug("saved model to {}".format(filepath))

        def on_train_end(self, logs=None):
            filepath = self.filepath_template.format(epoch="end", **logs)
            self.model.save(filepath, overwrite=True)
            self.v.logger.debug("saved model to {}".format(filepath))
