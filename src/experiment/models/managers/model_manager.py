from abc import ABC, abstractmethod
from typing import Tuple

import keras
import tensorflow as tf
from keras import Model
from keras.layers import Dense, Dropout
from tensorflow.python.layers.base import Layer

from constants import INPUT_SHAPE
from src.experiment.tasks.task_type import TaskType


class ModelManager(ABC):
    def __init__(self, export_path: str, create_task_model: bool, base_model_path: str = None,
                 base_model_weights: str = "imagenet",
                 base_model_pooling: str = "avg"):
        """
        Constructs manager responsible for creating, loading, and saving models.
        :param export_path: Where to save the model to.
        :param base_model_path: Path to load base model from.
        :param base_model_pooling: The pooling method for the base model if new model created.
        :param base_model_weights: The weights for the base model if new model created.
        """
        self.__model = None
        self.__feature_model = None
        self.activation = {}
        self.base_model_path = base_model_path
        self.export_path = export_path
        self.base_model_weights = base_model_weights
        self.base_model_pooling = base_model_pooling
        self.create_task_model = create_task_model
        print("Base Model Path:\t", self.base_model_path)
        print("Export path:\t\t", self.base_model_path)

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

    def create_model(self, task: Tuple[TaskType, str], n_outputs: int) -> Model:
        """
        Creates a Keras model.
        :return: The Keras model.
        """
        if self.__model is None:
            assert self.export_path is not None, "Export path is not set on model."
            task_name = self.__class__.__name__

            # 1. Create model
            if self.base_model_path:
                base_model = self.__load_previous_model(self.base_model_path)
            else:
                base_model = self.__create_pre_trained_model()

            if self.create_task_model:
                print("Model Modifications:", "new task head")
                output_layer = self.__add_task_head(base_model, task, n_outputs)
            else:
                print("Model Modifications:", "None.")
                output_layer = base_model.output

            model_name = "_".join([self.get_model_name(), task_name])
            model = tf.keras.Model(inputs=base_model.input, outputs=output_layer, name=model_name)
            self.__model = model
        return self.__model

    def __add_task_head(self, base_model: Model, task: Tuple[TaskType, str], n_outputs: int):
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

    def __load_previous_model(self, model_path: str) -> Model:
        """
        Loads model at given model path with autoloader.
        :param model_path: The path to the model weights and configuration.
        :return: Keras.Model
        """
        previous_model: Model = tf.keras.models.load_model(self.base_model_path)
        if self.create_task_model:
            return keras.Model(inputs=previous_model.input, outputs=previous_model.layers[-2].output)
        return previous_model

    def __create_pre_trained_model(self) -> Model:
        """
        Creates pre-trained model as a base model for task.
        :return:
        """
        base_model_class = self.get_model_constructor()
        return base_model_class(
            pooling=self.base_model_pooling,
            include_top=False,
            input_shape=INPUT_SHAPE,
            weights=self.base_model_weights,
            input_tensor=tf.keras.Input(shape=INPUT_SHAPE)
        )

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

    def get_feature_weights(self):
        """
        Gets parameters for feature layer.
        :return: Feature layer parameters
        """
        return self.get_model().layers[-1].get_weights()[0]
