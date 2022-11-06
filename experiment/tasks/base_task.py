import logging.config
from abc import ABC, abstractmethod
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.losses import mean_absolute_error
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from tensorflow.python.data import Dataset

from experiment.Food2Index import Food2Index
from experiment.models.base_models import BASE_MODELS, BaseModel, PRE_TRAINED_MODELS
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


class Task:
    def __init__(self,
                 base_model: str,
                 n_outputs=1,
                 n_epochs=N_EPOCHS,
                 load_weights=True,
                 load_on_init=True):
        section_heading = "Run Settings"
        logger.info(format_header(section_heading))
        logger.info("Task: " + self.__class__.__name__)
        self.base_model = base_model
        self.load_weights = load_weights
        self.n_outputs = n_outputs
        self.n_epochs = n_epochs
        self.checkpoint_path = get_checkpoint_path(self.__class__.__name__, base_model)
        self.model = self.make_model(base_model,
                                     n_outputs,
                                     self.__class__.__name__,
                                     base_model in PRE_TRAINED_MODELS)
        if load_on_init:
            self.load_model()
        logger.info(get_section_break(section_heading))

    @property
    @abstractmethod
    def loss_function(self):
        pass

    @property
    @abstractmethod
    def metric(self):
        pass

    @property
    @abstractmethod
    def task_type(self):
        pass

    @property
    @abstractmethod
    def task_mode(self):
        pass

    @abstractmethod
    def get_eval_dataset(self, name: str) -> [str]:
        pass

    @abstractmethod
    def get_training_data(self):
        pass

    @abstractmethod
    def get_validation_data(self):
        pass

    @abstractmethod
    def get_test_data(self):
        pass

    @abstractmethod
    def eval(self, dataset):
        pass

    def make_model(self, base_model_name, n_outputs: int, task_name, pre_trained_model=True):
        base_model_class = BASE_MODELS[base_model_name]
        if pre_trained_model:
            inputs = tf.keras.Input(shape=INPUT_SHAPE)
            base_model = base_model_class(
                pooling='avg',
                include_top=False,
                input_shape=INPUT_SHAPE,
                weights="imagenet",
                input_tensor=inputs
            )
            for layer in base_model.layers:
                layer._name = "_".join([layer.name, base_model_name, task_name])
        else:
            base_model = base_model_class()
        output_layer_name = "_".join(["output", base_model_name, task_name])
        model_name = "_".join([base_model_name, task_name])
        activation_function = None
        x = base_model.output
        if self.task_mode == TaskType.CLASSIFICATION:
            activation_function = "softmax"
            x = GlobalAveragePooling2D()(x)
            x = Dense(2048, activation='relu')(x)
            x = Dropout(0.5)(x)
        output_layer = tf.keras.layers.Dense(n_outputs,
                                             name=output_layer_name,
                                             activation=activation_function)(x)
        model = tf.keras.Model(inputs=base_model.input, outputs=output_layer, name=model_name)
        return model

    def load_model(self):
        if self.load_weights and os.path.isdir(os.path.dirname(self.checkpoint_path)):
            self.model = tf.keras.models.load_model(self.checkpoint_path)
            weights = "Previous best on validation"
        else:
            self.model.compile(optimizer="adam", loss=self.loss_function, metrics=[self.metric])
            weights = "Random"
        logger.info(format_name_val_info("Model", self.base_model))
        logger.info(format_name_val_info("Weights", weights))

    def train(self):
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

    def get_predictions(self, data):
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
    def initialize_dict_entry(dict_, key, init_val=0):
        if key not in dict_:
            dict_[key] = init_val
        return dict_, key

    @staticmethod
    def increment_dict_entry(dict_, key, child_key=None):
        dict_, key = Task.initialize_dict_entry(dict_, key, init_val={} if child_key else 0)
        if child_key:
            dict_, key = Task.initialize_dict_entry(dict_[key], child_key)
        dict_[key] += 1

    @staticmethod
    def combine_datasets(datasets):
        dataset = datasets[0].get_dataset(shuffle=False)
        image_count = len(datasets[0].get_image_paths())
        for d in datasets[1:]:
            dataset = dataset.concatenate(d.get_dataset(shuffle=False))
            image_count += len(d.get_image_paths())
        return dataset, image_count

    @staticmethod
    def print_datasets(datasets, image_count):
        print("Datasets:", ", ".join([d.__class__.__name__ for d in datasets]))
        print("Image Count:", image_count)


class RegressionTask(Task, ABC):
    task_type = TaskType.REGRESSION
    loss_function = "mse"
    metric = "mae"
    task_mode = "min"

    def __init__(self,
                 base_model: BaseModel,
                 n_outputs=1,
                 n_epochs=N_EPOCHS,
                 load_weights=True,
                 load_on_init=True):
        super().__init__(base_model,
                         n_outputs,
                         n_epochs,
                         load_weights=load_weights,
                         load_on_init=load_on_init)

    def eval(self, _):
        y_true, y_pred = self.get_predictions(self.get_test_data())
        mae = mean_absolute_error(y_true.flatten(), y_pred.flatten()).numpy()
        logger.info(format_name_val_info("Test Mean Absolute Error", mae))

    def get_predictions(self, data):
        y_true, y_pred = super().get_predictions(data)
        y_true = list(map(lambda v: v.numpy(), y_true))  # unpacks 1D vector into single number
        y_true = np.array(y_true)
        return y_true, y_pred


class ClassificationTask(Task, ABC):
    task_type = TaskType.CLASSIFICATION
    loss_function = tf.keras.losses.CategoricalCrossentropy()
    metric = "accuracy"
    task_mode = "max"

    def __init__(self,
                 base_model: BaseModel,
                 n_outputs=1,
                 n_epochs=N_EPOCHS,
                 load_weights=True,
                 load_on_init=True):
        super().__init__(base_model, n_outputs, n_epochs, load_weights, load_on_init)

    def eval(self, dataset_name=None):
        logger.info(format_header("Eval"))
        data = self.get_test_data() if dataset_name is None else self.get_eval_dataset(dataset_name)
        food2index = Food2Index()

        y_test, y_pred = self.get_predictions(data)  # no validation data on any class. task

        predictions = []
        labels = []

        class_tp = {}
        class_fp = {}
        class_fn = {}

        for test_vector, pred_vector in zip(y_test, y_pred):
            pred = np.argmax(pred_vector)
            label = np.argmax(test_vector)

            pred_name = food2index.get_ingredient(pred)
            label_name = food2index.get_ingredient(label)

            predictions.append(pred_name)
            labels.append(label_name)

            if pred == label:
                self.increment_dict_entry(class_tp, label_name)
            else:
                self.increment_dict_entry(class_fp, pred_name, label_name)
                self.increment_dict_entry(class_fn, label_name, pred_name)

        self.print_metrics(labels, predictions)
        logger.info(format_eval_results(class_tp, "TP"))
        logger.info(format_eval_results(class_fp, "FP"))
        logger.info(format_eval_results(class_fn, "FN"))

    @staticmethod
    def print_metrics(labels, predictions):
        matrix = confusion_matrix(labels, predictions)
        FP = matrix.sum(axis=0) - np.diag(matrix)
        FN = matrix.sum(axis=1) - np.diag(matrix)
        TP = np.diag(matrix)
        TN = matrix.sum() - (FP + FN + TP)
        logger.info("Predictions: %s" % len(predictions))
        logger.info("False Positive: %s" % FP.sum())
        logger.info("False Negatives: %s" % FN.sum())
        logger.info("True Positive: %s" % TP.sum())
        logger.info("True Negative: %s" % TN.sum())
