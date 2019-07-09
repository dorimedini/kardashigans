import keras
import utils as U
from verbose import Verbose


class ResourceManager(Verbose):
    """ Handles saving / loading trained models """
    def __init__(self, model_save_dir, model_load_dir, verbose=False):
        """
        :param model_save_dir: All saved models will be stored at this
            directory, under a subdirectory named by the date / time of
            initialization of the resource manager.
        :param model_load_dir: All loaded models are loaded from this
            directory. Loaded models must be directly contained in this
            path.
        """
        super(ResourceManager, self).__init__(verbose=verbose)
        self._model_save_dir = self._add_slash(model_save_dir)
        self._model_load_dir = self._add_slash(model_load_dir)
        self._model_file_template = "{model_name}.h5"

    def _add_slash(self, path_to_dir):
        if path_to_dir[-1] not in ['/', '\\']:
            path_to_dir += "/"
        return path_to_dir

    def _get_checkpoint_model_name(self, model_name, epoch):
        return model_name + "_epoch_{}".format(epoch)

    def _get_checkpoint_file_template(self, model_name):
        return model_name + "_epoch_{epoch}.h5"

    def get_epoch_save_period(self):
        return [0, 1, 2, 3, 8, 40, 90]

    def get_checkpoint_epoch_keys(self):
        return ['start'] + self.get_epoch_save_period() + ['end']

    def get_epoch_save_callback(self, model_name):
        filepath_template = self._model_save_dir + self._get_checkpoint_file_template(model_name)
        return SaveModelAtEpochsCallback(filepath_template=filepath_template,
                                         period=self.get_epoch_save_period(),
                                         verbose=self._verbose)

    def save_model(self, model, model_name):
        model.save(self._model_save_dir + self._model_file_template.format(model_name=model_name),
                   overwrite=True)

    def load_model(self, model_name):
        """ Uses the load_dir as base path """
        return keras.models.load_model(self._model_load_dir + self._model_file_template.format(model_name=model_name))

    def try_load_model(self, model_name):
        """
        If we're unsure the model exists but we'd like to use it if it is,
        call this method (returns None if model couldn't be loaded)
        """
        try:
            return self.load_model(model_name)
        except:
            self._print("Couldn't load model {}".format(model_name))
        return None

    def try_load_model_checkpoints(self, model_name, period):
        """
        Loads saved models at specific epochs. Intended for use with
        get_epoch_save_callback (used by Experiment objects via the
        Trainer's set_checkpoint_callbacks method).

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
        start_model = self.try_load_model(self._get_checkpoint_model_name(model_name, 'start'))
        end_model = self.try_load_model(self._get_checkpoint_model_name(model_name, 'end'))
        if start_model:
            epoch_model_map['start'] = start_model
        else:
            self._print("No start epoch found (tried {})".format(start_model))
        if end_model:
            epoch_model_map['end'] = end_model
        else:
            self._print("No end epoch found (tried {})".format(end_model))
        for epoch in period:
            epoch_model = self.try_load_model(self._get_checkpoint_model_name(model_name, epoch))
            if epoch_model:
                epoch_model_map[epoch] = epoch_model
            else:
                self._print("Saved epoch {} not found at {}".format(epoch, self._get_checkpoint_model_name(model_name, epoch)))
        return epoch_model_map


class SaveModelAtEpochsCallback(Callback):
    def __init__(self, filepath_template, period=[], verbose=False):
        super(SaveModelAtEpochsCallback, self).__init__()
        self.filepath_template = filepath_template
        self.period = period
        self._printer = Verbose(verbose=verbose)

    def on_train_begin(self, logs=None):
        filepath = self.filepath_template.format(epoch="start", **logs)
        self.model.save(filepath, overwrite=True)
        self._printer._print("saved model to {}".format(filepath))

    def on_epoch_end(self, epoch, logs=None):
        if epoch in self.period:
            filepath = self.filepath_template.format(epoch=epoch, **logs)
            self.model.save(filepath, overwrite=True)
            self._printer._print("saved model to {}".format(filepath))

    def on_train_end(self, logs=None):
        filepath = self.filepath_template.format(epoch="end", **logs)
        self.model.save(filepath, overwrite=True)
        self._printer._print("saved model to {}".format(filepath))
