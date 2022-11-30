from abc import ABC, abstractmethod
from typing import Callable, Tuple

import keras
import tensorflow as tf
from keras import Model
from keras.layers import Dense, Dropout
from tensorflow.python.layers.base import Layer

from constants import INPUT_SHAPE
from src.experiment.models.checkpoint_creator import get_checkpoint_path
from src.experiment.tasks.task_type import TaskType


class ModelManager(ABC):
    def __init__(self, model_path: str = None, export_path: str = None):
        """
        Constructs manager responsible for creating, loading, and saving models.
        :param model_path:
        :type model_path:
        :param export_path:
        :type export_path:
        """
        self.__model = None
        self.__feature_model = None
        self.activation = {}
        self.model_path = model_path
        self.export_path = export_path

    def set_task_checkpoint(self, task: object, checkpoint_name: str = None) -> None:
        """
        Sets the export path of the manager to the checkpoint path with model sub-folder.
        :param task: The task whose checkpoint path is used to export to.
        :param checkpoint_name: The optional name of the sub-folder with the task checkpoint path.
        :return: None
        """
        if self.export_path is None:
            self.export_path = get_checkpoint_path(task.__class__.__name__, self.get_model_name())
        print("Export path is:\t", self.export_path)

    def get_model(self) -> Model:
        """
        :return: Returns the constructed model.
        """
        if self.__model is None:
            raise Exception("Model has not been constructed.")
        return self.__model

    def save_model(self):
        """
        Saves current model to export path.
        :return: None
        """
        model = self.get_model()
        model.save(self.export_path)
        print("Model saved!", "(", self.export_path, ")")

    def get_feature_model(self) -> tf.keras.Model:
        """
        Returns new model that outputs feature map in addition to model output.
        :return: New keras model reference base model.
        """
        if self.__feature_model is None:
            model = self.get_model()
            layer_outputs = self.get_feature_layer(model).output
            self.__feature_model = Model(inputs=model.input, outputs=[model.output, layer_outputs])
        return self.__feature_model

    def create_model(self, task: Tuple[TaskType, str], n_outputs: int, pre_trained_model=True) -> Model:
        """
        Creates a Keras model.
        :return: The Keras model.
        """
        if self.__model is None:
            assert self.export_path is not None, "Export path is not set on model."
            task_name = self.__class__.__name__

            # 1. Create model
            base_model = self.__create_base_model(pre_trained_model)
            output_layer = self.__add_output_head(base_model, task, n_outputs)

            model_name = "_".join([self.get_model_name(), task_name])
            model = tf.keras.Model(inputs=base_model.input, outputs=output_layer, name=model_name)

            self.__model = model
        return self.__model

    def __create_base_model(self, is_pretrained: bool):
        """
        Creates the base model for given task.
        :param is_pretrained: Whether to load image_name pre-trained weights.
        :rtype:
        """
        base_model_class = self.get_model_constructor()
        if self.model_path:
            return self.load_previous_model(self.model_path, remove_last_layer=True)
        if is_pretrained:
            return self.create_pre_trained_model(base_model_class)
        return base_model_class()

    def __add_output_head(self, base_model: Model, task: Tuple[TaskType, str], n_outputs: int):
        """
        Adds task-specific set of layers to base model.
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

    def load_previous_model(self, model_path: str, remove_last_layer: bool = True) -> Model:
        """
        Loads model at given model path with autoloader.
        :param model_path: The path to the model weights and configuration.
        :param remove_last_layer: Whether to remove the last layer of the model.
        :return: Keras.Model
        """
        print("Loading model checkpoint:", model_path)
        previous_model: Model = tf.keras.models.load_model(self.model_path)
        if remove_last_layer:
            return keras.Model(inputs=previous_model.input, outputs=previous_model.layers[-2].output)
        return previous_model

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
