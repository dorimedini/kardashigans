from kardashigans.experiment import Experiment
from kardashigans.simple_experiment import SimpleExperiment
from kardashigans.trainer import FCTrainer
from keras.datasets import mnist, cifar10
from keras import optimizers


class Baseline(SimpleExperiment):
    def __init__(self, root_dir, **kwargs):
        mnist_name = Baseline.get_model_name(Experiment.get_dataset_name(mnist))
        cifar10_name = Baseline.get_model_name(Experiment.get_dataset_name(cifar10))
        super(Baseline, self).__init__(name='Baseline',
                                       model_names=[mnist_name, cifar10_name],
                                       trainers={
                                           mnist_name: Baseline.construct_dataset_trainer(mnist),
                                           cifar10_name: Baseline.construct_dataset_trainer(cifar10)
                                       },
                                       root_dir=root_dir,
                                       **kwargs)

    @staticmethod
    def get_model_name(dataset_name):
        return "{dataset}_fc{layers}_batch{batch}_epochs{epochs}_{opt}" \
               "".format(dataset=dataset_name,
                         layers=3,
                         batch=32,
                         epochs=100,
                         opt='SGD_moment0.9_NesterovTRUE')

    @staticmethod
    def construct_dataset_trainer(dataset):
        return FCTrainer(dataset=dataset,
                         epochs=100,
                         n_layers=3,
                         batch_size=32,
                         optimizer=optimizers.SGD(momentum=0.9, nesterov=True))

    def get_layer_indices_list(self, trainer):
        """ We want to check robustness of each individual layer by itself """
        return [[layer] for layer in range(trainer.get_n_parameter_layers())]
