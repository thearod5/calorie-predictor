from abc import ABC, abstractmethod
from typing import Callable, Tuple

import tensorflow as tf
from keras import Model
from keras.layers import Dense, Dropout
from tensorflow.python.layers.base import Layer

from constants import INPUT_SHAPE
from src.experiment.tasks.task_type import TaskType


class ModelManager(ABC):
    def __init__(self):
        self.__model = None
        self.activation = {}

    def create_feature_model(self) -> tf.keras.Model:
        """
        Returns new model that outputs feature map in addition to model output.
        :return: New keras model reference base model.
        """
        model = self.get_model()
        layer_outputs = self.get_feature_layer(model).output
        return Model(inputs=model.input, outputs=[model.output, layer_outputs])

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

    def create_model(self, task: Tuple[TaskType, str], n_outputs: int, pre_trained_model=True) -> Model:
        """
        Creates a Keras model.
        :return: The Keras model.
        """
        if self.__model is None:
            task_name = self.__class__.__name__

            # 1. Create model
            base_model = self.__create_base_model(pre_trained_model)
            model_name = "_".join([self.get_model_name(), task_name])
            output_layer = self.__add_output_head(base_model, task, n_outputs)
            model = tf.keras.Model(inputs=base_model.input, outputs=output_layer, name=model_name)

            self.__model = model
        return self.__model

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

    @staticmethod
    def create_pre_trained_model(base_model_class: Callable[..., Model], weights: str = "imagenet", include_top=False,
                                 pooling: str = "avg") -> Model:
        """
        Creates pre-trained model for task.
        :param include_top: Whether to include classification top.
        :param base_model_class: The base model defining its architecture.
        :param weights: The pre-trained weights.
        :param pooling: The pooling function.
        :return:
        """
        inputs = tf.keras.Input(shape=INPUT_SHAPE)
        print("Initializing with:", weights)
        print("Include top:", include_top)
        base_model_obj = base_model_class(
            pooling=pooling,
            include_top=include_top,
            input_shape=INPUT_SHAPE,
            weights=weights,
            input_tensor=inputs
        )

        return base_model_obj

    def __add_output_head(self, base_model: Model, task: Tuple[TaskType, str], n_outputs: int):
        """
        Adds task-specific set of layers to base model.
        TODO: Infer n_outputs by task
        :param base_model: The base model to attach layers to.
        :param n_outputs: The final number of outputs.
        :param task: The task being performed
        :return:
        """
        output_layer_name = "_".join(["output", self.get_model_name(), task[1]])
        output_layer = base_model.output
        if task[0] == TaskType.CLASSIFICATION:
            output_activation = "softmax"
            output_layer = self.append_hidden_layer(base_model)
        else:
            output_activation = "linear"
        return tf.keras.layers.Dense(n_outputs,
                                     name=output_layer_name,
                                     activation=output_activation)(output_layer)

    @staticmethod
    def append_hidden_layer(base_model, hidden_size: int = 2048, activation: str = "relu",
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
        x = Dense(hidden_size, activation=activation)(x)
        x = Dropout(dropout)(x)

        return x

    @abstractmethod
    def get_model_name(self) -> str:
        """
        :return: The name of the model.
        """
        pass

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
    def get_feature_weights(self):
        """
        Gets parameters for feature layer.
        :return: Feature layer parameters
        """
        pass