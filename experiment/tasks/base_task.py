import logging.config
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.losses import mean_absolute_error
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.python.data import Dataset

from datasets.abstract_dataset import AbstractDataset
from experiment.Food2Index import Food2Index
from experiment.models.checkpoint_creator import get_checkpoint_path
from experiment.models.managers.model_manager import ModelManager
from experiment.models.managers.model_managers import PRE_TRAINED_MODELS
from experiment.tasks.task_mode import TaskMode
from experiment.tasks.task_type import TaskType
from logging_utils.utils import *

logging.config.fileConfig(LOG_CONFIG_FILE)
logger = logging.getLogger()


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


class AbstractTask(ABC):
    def __init__(self, model_manager: ModelManager, n_outputs: int = 1, n_epochs: int = N_EPOCHS,
                 load_weights: bool = True,
                 load_on_init: bool = True):
        """
        Represents a task for the model
        :param model_manager: the model to use for the task
        :param n_outputs: the number of nodes for the output layer
        :param n_epochs: the number of epochs to run training for
        :param load_weights: if True, loads existing weights
        :param load_on_init: if True, loads the model in task __init__
        """
        section_heading = "Run Settings"
        task_name = self.__class__.__name__
        logger.info(format_header(section_heading))
        logger.info("Task: " + task_name)

        self.load_weights = load_weights
        self.n_outputs = n_outputs
        self.n_epochs = n_epochs

        self.model_manager = model_manager
        is_pretrained = any([isinstance(model_manager, pt.value) for pt in PRE_TRAINED_MODELS])
        self.model = model_manager.create_model((self.task_mode, task_name), n_outputs=n_outputs,
                                                pre_trained_model=is_pretrained)
        self.checkpoint_path = get_checkpoint_path(self.__class__.__name__, self.model_manager.model_name.lower())
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
        logger.info(format_name_val_info("Model", self.model_manager.model_name))
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
        dict_ = AbstractTask.initialize_dict_entry(dict_, key, init_val={} if child_key else 0)
        if child_key:
            dict_ = AbstractTask.initialize_dict_entry(dict_[key], child_key)
        dict_[key] += 1

    @staticmethod
    def combine_datasets(datasets: List[AbstractDataset]) -> Tuple[AbstractDataset, int]:
        """
        Combines the datasets in the list
        :param datasets: a list of a datasets
        :return: the new combined dataset and the total image count after combining
        """
        dataset = datasets[0].get_dataset(shuffle=False)
        image_count = len(datasets[0].get_image_paths(datasets[0].image_dir))
        for d in datasets[1:]:
            dataset = dataset.concatenate(d.get_dataset(shuffle=False))
            image_count += len(d.get_image_paths(d.image_dir))
        return dataset, image_count

    @staticmethod
    def print_datasets(datasets: List[AbstractDataset], image_count: int):
        """
        Prints the names of all datasets in the list and the total image count
        :param datasets: a list of datasets
        :param image_count: the image count
        :return: None
        """
        print("Datasets:", ", ".join([d.__class__.__name__ for d in datasets]))
        print("Image Count:", image_count)
