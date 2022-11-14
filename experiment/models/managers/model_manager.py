from abc import ABC, abstractmethod
from typing import Callable, Tuple

import tensorflow as tf
from keras import Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.python.layers.base import Layer

from constants import INPUT_SHAPE
from experiment.tasks.task_mode import TaskMode
from experiment.tasks.task_type import TaskType


class ModelManager(ABC):
    def __init__(self):
        self.__model = None
        self.activation = {}
        self.model_name = self.get_model_constructor().__class__.__name__

    @abstractmethod
    def get_model_constructor(self):
        """
        Returns the constructor to initialize model.
        :return:
        """
        pass

    @staticmethod
    @abstractmethod
    def get_feature_layer(model: Model) -> Layer:
        """
        Returns the feature layer of the model.
        :return: The feature layer.
        """
        pass

    @abstractmethod
    def get_parameters(self):
        """
        Gets parameters for feature layer.
        :return: Feature layer parameters
        """
        pass

    def get_model(self) -> Model:
        if self.__model is None:
            raise Exception("Model has not been constructed.")
        return self.__model

    def get_features(self):
        """
        Returns the features associated with the last run on the model
        :return: The output of the feature layer of layer run.
        """
        return self.activation["features"]

    def create_model(self, task: Tuple[TaskMode, str], n_outputs: int, pre_trained_model=True) -> Model:
        """
        Creates a Keras model.
        :return: The Keras model.
        """
        if self.__model is None:
            task_name = self.__class__.__name__

            # 1. Create model
            base_model = self.__create_base_model(pre_trained_model)
            model_name = "_".join([self.model_name, task_name])
            output_layer = self.__add_output_head(base_model, task, n_outputs)
            model = tf.keras.Model(inputs=base_model.input, outputs=output_layer, name=model_name)

            # 2. Register feature hook
            # feature_layer = self._get_feature_layer(model)
            # feature_layer.register_forward_hook(self.__create_feature_hook("features"))

            # Given layers readable names
            # for layer in base_model.layers:
            #     layer._name = "_".join([layer.name, self.model_name, task_name])

            self.__model = model
        return self.__model

    @staticmethod
    def append_dense_layer(base_model, hidden_size: int = 2048, activation: str = "relu",
                           dropout: float = 0.5) -> Model:
        """
        Attaches hidden layer with dropout to model.
        :param base_model: The model to append layer to.
        :param hidden_size: The number of neurons in hidden layer.
        :param activation: The activation function to apply to dense layer.
        :param dropout: The probability a neuron weight will be dropped.
        :return: Model with layer attached.
        """
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(hidden_size, activation=activation)(x)
        x = Dropout(dropout)(x)

        return x

    @staticmethod
    def create_pre_trained_model(base_model_class: Callable[..., Model], weights: str = "imagenet", include_top=False,
                                 pooling: str = "avg") -> Model:
        """
        Creates pre-trained model for task.
        :param base_model_class: The base model defining its architecture.
        :param weights: The pre-trained weights.
        :param pooling: The pooling function.
        :return:
        """
        inputs = tf.keras.Input(shape=INPUT_SHAPE)
        base_model_obj = base_model_class(
            pooling=pooling,
            include_top=include_top,
            input_shape=INPUT_SHAPE,
            weights=weights,
            input_tensor=inputs
        )

        return base_model_obj

    def __create_base_model(self, is_pretrained: bool):
        """
        Creates the base model for given task.
        :param is_pretrained: Whether to load image_namepre-trained weights.
        :return:
        :rtype:
        """
        base_model_class = self.get_model_constructor()
        if is_pretrained:
            base_model_obj = self.create_pre_trained_model(base_model_class)
        else:
            base_model_obj = base_model_class()
        return base_model_obj

    def __add_output_head(self, base_model: Model, task: Tuple[TaskMode, str], n_outputs: int,
                          output_activation="softmax"):
        """
        Adds task-specific set of layers to base model.
        TODO: Infer n_outputs by task
        :param base_model: The base model to attach layers to.
        :param n_outputs: The final number of outputs.
        :param task: The task being performed
        :param output_activation: The activation function to use on the last layer.
        :return:
        """
        output_layer_name = "_".join(["output", self.model_name, task[1]])
        output_layer = base_model.output
        if task[0] == TaskType.CLASSIFICATION:
            output_layer = self.append_dense_layer(base_model)
        return tf.keras.layers.Dense(n_outputs,
                                     name=output_layer_name,
                                     activation=output_activation)(output_layer)

    def __create_feature_hook(self, hook_name: str):
        """
        Creates hook that will store output at given feature name.
        :param hook_name: The name of the feature.
        :return: The hook that stores output.
        """

        def hook(_1, _2, output):
            self.activation[hook_name] = output

        return hook
