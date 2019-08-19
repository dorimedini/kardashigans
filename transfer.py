from kardashigans.analyze_model import AnalyzeModel
from kardashigans.experiment import Experiment
from random import sample


class SplitDataset:
    """
    Dataset with samples corresponding to certain tagging labels only
    """

    def __init__(self, dataset, labels):
        (x_train, y_train), (x_test, y_test) = dataset.load_data()
        self._x_train, self._y_train = self.split_set(x_train, y_train, labels)
        self._x_test, self._y_test = self.split_set(x_test, y_test, labels)

    @staticmethod
    def split_set(x, y, labels):
        indices = [i for i in range(y.shape[0]) if y[i] in labels]
        return x[indices, ...], y[indices, ...]

    def load_data(self):
        return (self._x_train, self._y_train), (self._x_test, self._y_test)


class TransferExperiment(Experiment):
    """
    replicated the method demonstrated in "How transferable are features in deep neural networks?"
    supports only trainers with "layers_to_freeze" and "weight_map" arguments
    """

    def __init__(self, dataset, trainer_class, trainer_kwargs, n_classes,
                 split_labels=None, model_a_key="a", model_b_key="b", *args, **kwargs):
        if split_labels is None:
            split_labels = sample(list(range(n_classes)), n_classes // 2)
        data_a = SplitDataset(dataset, split_labels)
        data_b = SplitDataset(dataset, [i for i in range(n_classes) if i not in split_labels])
        trainer_a = trainer_class(dataset=data_a, **trainer_kwargs)
        trainer_b = trainer_class(dataset=data_b, **trainer_kwargs)
        super(TransferExperiment, self).__init__(model_names=[model_a_key, model_b_key],
                                                 trainers={
                                                     model_a_key: trainer_a,
                                                     model_b_key: trainer_b
                                                 },
                                                 *args, **kwargs)
        self._model_a_key = model_a_key
        self._model_b_key = model_b_key
        self._trainer_class = trainer_class
        if "layers_to_freeze" in trainer_kwargs:
            trainer_kwargs.pop("layers_to_freeze")
        if "weight_map" in trainer_kwargs:
            trainer_kwargs.pop("weight_map")
        self._trainer_kwargs = trainer_kwargs
        self._data_a = data_a
        self._data_b = data_b

    @staticmethod
    def create_name_from_list(base, layers_to_copy):
        if not layers_to_copy:
            return base
        else:
            return base + str(layers_to_copy).strip("[ ]")

    def get_model_acc(self, model_name, x_test, y_test):
        with self.open_model(model_name) as model:
            eval_results = model.evaluate(x_test, y_test)
            eval_acc = eval_results[model.metrics_names.index('acc')]
        return eval_acc

    def get_model_weights(self, model_name):
        with self.open_model(model_name) as model:
            weights = {i: model.layers[i].get_weights() for i in self._trainers[model_name].get_weighted_layers_indices()}
        return weights

    def go(self, layers_to_copy_set, results_name="transfer_results", base_name="base"):
        weights_a = self.get_model_weights(self._model_a_key)
        weights_b = self.get_model_weights(self._model_b_key)
        x_test_a, y_test_a = self.get_test_data(self._model_a_key)
        x_test_b, y_test_b = self.get_test_data(self._model_b_key)
        results = self._resource_manager.get_existing_results(self._name, results_name)
        if base_name not in results:
            results = {base_name: {self._model_a_key: self.get_model_acc(self._model_a_key, x_test_a, y_test_a),
                                self._model_b_key: self.get_model_acc(self._model_a_key, x_test_b, y_test_b)}}
            self._resource_manager.save_results(results, self._name, results_name)
        for layers_to_copy in layers_to_copy_set:
            if self.create_name_from_list("", layers_to_copy) not in results:
                ab_name = self.create_name_from_list("ab", layers_to_copy)
                abp_name = self.create_name_from_list("abp", layers_to_copy)
                ba_name = self.create_name_from_list("ba", layers_to_copy)
                bap_name = self.create_name_from_list("bap", layers_to_copy)
                curr_weights_a = {i: weights_a[i] for i in weights_a if i in layers_to_copy}
                curr_weights_b = {i: weights_b[i] for i in weights_b if i in layers_to_copy}
                ab_trainer = self._trainer_class(dataset=self._data_b, layers_to_freeze=layers_to_copy,
                                                 weight_map=curr_weights_a, **self._trainer_kwargs)
                abp_trainer = self._trainer_class(dataset=self._data_b,
                                                  weight_map=curr_weights_a, **self._trainer_kwargs)
                ba_trainer = self._trainer_class(dataset=self._data_a, layers_to_freeze=layers_to_copy,
                                                 weight_map=curr_weights_b, **self._trainer_kwargs)
                bap_trainer = self._trainer_class(dataset=self._data_a,
                                                  weight_map=curr_weights_b, **self._trainer_kwargs)
                curr_trainers = {ab_name: ab_trainer,
                                 ba_name: ba_trainer,
                                 abp_name: abp_trainer,
                                 bap_name: bap_trainer}
                model_names = list(curr_trainers.keys())
                self._model_names.extend(model_names)
                self._trainers.update(curr_trainers)

                curr_results = {
                    "ab": self.get_model_acc(ab_name, x_test_b, y_test_b),
                    "abp": self.get_model_acc(abp_name, x_test_b, y_test_b),
                    "ba": self.get_model_acc(ba_name, x_test_a, y_test_a),
                    "bap": self.get_model_acc(bap_name, x_test_a, y_test_a),
                    }
                results.update({self.create_name_from_list("", layers_to_copy): curr_results})
                self._resource_manager.save_results(results, self._name, results_name)

        AnalyzeModel.generate_transfer_graph(results=results, output_dir=self._output_dir)
