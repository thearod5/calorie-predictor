import logging.config
from abc import abstractmethod
from enum import Enum
from typing import Tuple, List, Any, Dict

import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.losses import mean_absolute_error
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from tensorflow.python.data import Dataset
import keras

from experiment.Food2Index import Food2Index
from experiment.models.model_identifiers import BaseModel, PRE_TRAINED_MODELS
from experiment.models.checkpoint_creator import get_checkpoint_path
from logging_utils.utils import *

logging.config.fileConfig(LOG_CONFIG_FILE)
logger = logging.getLogger()


class TaskMode(Enum):
    TRAIN = "TRAIN"
    EVAL = "EVAL"


class TaskType(Enum):
    REGRESSION = "REGRESSION"
    CLASSIFICATION = "CLASSIFICATION"


def sample_data(data: Dataset):
    ingredient_index_map = Food2Index()
    for i, (image, label) in enumerate(data.take(9)):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image[0, :, :, :])
        plt.title(" ".join(ingredient_index_map.to_ingredients_list(label[0])))
        plt.axis("off")


augmentation_generator = ImageDataGenerator(rotation_range=15,
                                            width_shift_range=0.1,
                                            height_shift_range=0.1,
                                            shear_range=0.01,
                                            zoom_range=[0.9, 1.25],
                                            horizontal_flip=True,
                                            vertical_flip=False,
                                            fill_mode='reflect',
                                            data_format='channels_last',
                                            brightness_range=[0.5, 1.5])

class BaseTask:
    def __init__(self, base_model: BaseModel, n_outputs: int = 1, n_epochs: int = N_EPOCHS, load_weights: bool = True,
                 load_on_init: bool = True):
        """
        Represents a task for the model
        :param base_model: the model to use for the task
        :param n_outputs: the number of nodes for the output layer
        :param n_epochs: the number of epochs to run training for
        :param load_weights: if True, loads existing weights
        :param load_on_init: if True, loads the model in task __init__
        """
        section_heading = "Run Settings"
        logger.info(format_header(section_heading))
        logger.info("Task: " + self.__class__.__name__)
        self.base_model = base_model
        self.load_weights = load_weights
        self.n_outputs = n_outputs
        self.n_epochs = n_epochs
        self.checkpoint_path = get_checkpoint_path(self.__class__.__name__, base_model.name.lower())
        self.model = self.make_model(base_model,
                                     n_outputs,
                                     base_model in PRE_TRAINED_MODELS)
        if load_on_init:
            self.load_model()
        logger.info(get_section_break(section_heading))

    @property
    @abstractmethod
    def loss_function(self) -> keras.losses.Loss:
        """
        Defines the loss function for the model
        :return: the loss function
        """
        pass

    @property
    @abstractmethod
    def metric(self) -> str:
        """
        Defines the name of the metric to use for evaluating the model
        :return: the name of the metric
        """
        pass

    @property
    @abstractmethod
    def task_type(self) -> TaskType:
        """
        Defines the type of task (e.g. Classification or Regression)
        :return:
        """
        pass

    @property
    @abstractmethod
    def task_mode(self) -> TaskMode:
        """
        Defines the task mode (e.g. Train or Eval)
        :return: the task mode
        """
        pass

    @abstractmethod
    def get_eval_dataset(self, name: str) -> tf.data.Dataset:
        """
        Gets the dataset to use for evaluation
        :param name: the name of the dataset
        :return: the dataset
        """
        pass

    @abstractmethod
    def get_training_data(self) -> tf.data.Dataset:
        """
        Gets the dataset to use for training
        :return: the dataset
        """
        pass

    @abstractmethod
    def get_validation_data(self) -> tf.data.Dataset:
        """
        Gets the dataset to use for validation
        :return: the dataset
        """
        pass

    @abstractmethod
    def get_test_data(self) -> tf.data.Dataset:
        """
        Gets the dataset to use for testing
        :return: the dataset
        """
        pass

    @abstractmethod
    def eval(self, dataset_name: str):
        """
        Evaluates the model on the given dataset
        :param dataset_name: the name of the dataset to evaluate on
        :return: None
        """
        pass

    def make_model(self, base_model: BaseModel, n_outputs: int, pre_trained_model=True) -> tf.keras.Model:
        """
         Makes the task's model
         :param base_model: the model to use for the task
         :param n_outputs: the number of nodes for the output layer
         :param pre_trained_model: if True, assumes base_model is a pre_trained_model
         :return the model
         """
        base_model_class = base_model.value
        task_name = self.__class__.__name__,
        if pre_trained_model:
            inputs = tf.keras.Input(shape=INPUT_SHAPE)
            base_model_obj = base_model_class(
                pooling='avg',
                include_top=False,
                input_shape=INPUT_SHAPE,
                weights="imagenet",
                input_tensor=inputs
            )
            for layer in base_model_obj.layers:
                layer._name = "_".join([layer.name, base_model.name, task_name])
        else:
            base_model_obj = base_model_class()
        output_layer_name = "_".join(["output", base_model.name, task_name])
        model_name = "_".join([base_model.name, task_name])
        activation_function = None
        x = base_model_obj.output
        if self.task_mode == TaskType.CLASSIFICATION:
            activation_function = "softmax"
            x = GlobalAveragePooling2D()(x)
            x = Dense(2048, activation='relu')(x)
            x = Dropout(0.5)(x)
        output_layer = tf.keras.layers.Dense(n_outputs,
                                             name=output_layer_name,
                                             activation=activation_function)(x)
        model = tf.keras.Model(inputs=base_model_obj.input, outputs=output_layer, name=model_name)
        return model

    def load_model(self):
        """
        Handles loading the model weights and/or compiling
        :return: None
        """
        if self.load_weights and os.path.isdir(os.path.dirname(self.checkpoint_path)):
            self.model = tf.keras.models.load_model(self.checkpoint_path)
            weights = "Previous best on validation"
        else:
            self.model.compile(optimizer="adam", loss=self.loss_function, metrics=[self.metric])
            weights = "Random"
        logger.info(format_name_val_info("Model", self.base_model.name))
        logger.info(format_name_val_info("Weights", weights))

    def train(self):
        """
        Trains the model
        :return: None
        """
        logger.info(format_header("Training"))
        task_monitor = "val_" + self.metric
        model_checkpoint_callback = ModelCheckpoint(
            filepath=self.checkpoint_path,
            monitor=task_monitor,
            mode=self.task_mode,
            verbose=1,
            save_best_only=True)
        training_data = self.get_training_data()
        self.model.fit(self.augment_dataset(training_data),
                       epochs=self.n_epochs,
                       validation_data=self.get_validation_data(),
                       callbacks=[model_checkpoint_callback])

    def get_predictions(self, data) -> Tuple[List, Any]:
        """
        Gets the models predictions
        :param data: data to predict on
        :return: the expected results and those predicted
        """
        logger.info(format_header("Predicting"))
        y_pred = self.model.predict(data)
        y_true = [y_vector for _, batch_y in data for y_vector in batch_y]
        return y_true, y_pred

    @staticmethod
    def augment_dataset(in_gen):
        for in_x, in_y in in_gen:
            g_x = augmentation_generator.flow(in_x,
                                              in_y,
                                              batch_size=in_x.shape[0])
            x, y = next(g_x)

            yield x, y

    @staticmethod
    def initialize_dict_entry(dict_: dict, key: str, init_val=0) -> Dict:
        """
        Initializes a key in the dictionary the given init_val
        :param dict_: the dictionary
        :param key: the key to initialize inside the dictionary
        :param init_val: the value to initialize the entry to
        :return: the dictionary
        """
        if key not in dict_:
            dict_[key] = init_val
        return dict_

    @staticmethod
    def increment_dict_entry(dict_, key, child_key=None):
        """
        Increments the entry in a dictionary by 1
        :param dict_: the dictionary
        :param key: the key containing the value to increment
        :param child_key: if provided, increments the value inside of the child_key which is in the dictionary mapped to the key
        :return: None
        """
        dict_ = BaseTask.initialize_dict_entry(dict_, key, init_val={} if child_key else 0)
        if child_key:
            dict_ = BaseTask.initialize_dict_entry(dict_[key], child_key)
        dict_[key] += 1

    @staticmethod
    def combine_datasets(datasets: List[tf.data.AbstractDataset]) -> Tuple[tf.data.AbstractDataset, int]:
        """
        Combines the datasets in the list
        :param datasets: a list of a datasets
        :return: the new combined dataset and the total image count after combining
        """
        dataset = datasets[0].get_dataset(shuffle=False)
        image_count = len(datasets[0].get_image_paths())
        for d in datasets[1:]:
            dataset = dataset.concatenate(d.get_dataset(shuffle=False))
            image_count += len(d.get_image_paths())
        return dataset, image_count

    @staticmethod
    def print_datasets(datasets: List[tf.data.AbstractDataset], image_count: int):
        """
        Prints the names of all datasets in the list and the total image count
        :param datasets: a list of datasets
        :param image_count: the image count
        :return: None
        """
        print("Datasets:", ", ".join([d.__class__.__name__ for d in datasets]))
        print("Image Count:", image_count)
