import json
import os
import pprint
from abc import ABC, abstractmethod
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import mean_absolute_error
from tensorflow.python.data import Dataset

from constants import INPUT_SHAPE, N_EPOCHS, PROJECT_DIR
from experiment.Food2Index import Food2Index
from experiment.tasks.test_model import test_model


def convert_to_task(base_model_class, n_outputs: int):
    inputs = tf.keras.Input(shape=INPUT_SHAPE)
    base_model = base_model_class(
        pooling='avg',
        include_top=False,
        input_shape=INPUT_SHAPE,
        weights="imagenet",
        input_tensor=inputs
    )
    classification_layer = tf.keras.layers.Dense(n_outputs)(base_model.layers[-1].output)
    model = tf.keras.Model(inputs=base_model.input, outputs=classification_layer)
    return model


class TaskMode(Enum):
    TRAIN = "TRAIN"
    EVAL = "EVAL"


class TaskType(Enum):
    REGRESSION = "REGRESSION"
    CLASSIFICATION = "CLASSIFICATION"


class BaseModel(Enum):
    VGG = "vgg"
    RESNET = "resnet"
    XCEPTION = "xception"
    TEST = "test"


BASE_MODELS = {
    BaseModel.VGG: tf.keras.applications.VGG19,
    BaseModel.RESNET: tf.keras.applications.ResNet50,
    BaseModel.XCEPTION: tf.keras.applications.Xception,
    BaseModel.TEST: test_model

}


def create_checkpoint_path(task, base_model: BaseModel):
    task_name = task.__class__.__name__
    base_model_name = base_model.value
    return os.path.join(PROJECT_DIR, "results", "checkpoints", task_name, base_model_name, "cp.ckpt")


def sample_data(data: Dataset):
    ingredient_index_map = Food2Index()
    for i, (image, label) in enumerate(data.take(9)):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image[0, :, :, :])
        plt.title(" ".join(ingredient_index_map.to_ingredients_list(label[0])))
        plt.axis("off")


class Task:
    def __init__(self,
                 base_model: BaseModel,
                 task_type: TaskType,
                 n_outputs=1,
                 n_epochs=N_EPOCHS,
                 load_weights=True,
                 load_on_init=True):
        self.base_model = base_model
        self.task_type = task_type
        self.load_weights = load_weights
        self.n_outputs = n_outputs
        self.n_epochs = n_epochs
        self.checkpoint_path = create_checkpoint_path(self, base_model)
        self.model = convert_to_task(BASE_MODELS[base_model], n_outputs)
        self.is_regression = task_type == TaskType.REGRESSION
        if load_on_init:
            self.load_model()

    @property
    @abstractmethod
    def training_data(self):
        pass

    @property
    @abstractmethod
    def validation_data(self):
        pass

    @property
    @abstractmethod
    def test_data(self):
        pass

    @property
    @abstractmethod
    def eval(self):
        pass

    def load_model(self):
        if self.is_regression:
            self.model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        else:
            self.model.compile(optimizer="adam",
                               loss=tf.keras.losses.CategoricalCrossentropy(),
                               metrics=["accuracy"])

        if self.load_weights and os.path.isdir(os.path.dirname(self.checkpoint_path)):
            print("Loading previous weights...")
            self.model = tf.keras.models.load_model(self.checkpoint_path)
        print("Loading: " + self.base_model.value + " on " + self.__class__.__name__)

    def train(self):
        task_monitor = "val_mae" if self.is_regression else "val_accuracy"
        task_mode = "min" if self.is_regression else "max"
        model_checkpoint_callback = ModelCheckpoint(
            filepath=self.checkpoint_path,
            monitor=task_monitor,
            mode=task_mode,
            verbose=1,
            save_best_only=True)

        self.model.fit(self.training_data(),
                       epochs=self.n_epochs,
                       validation_data=self.validation_data(),
                       callbacks=[model_checkpoint_callback])

    def get_predictions(self, data):
        batch_predictions = self.model.predict(data)

        # TODO: Refactor
        y_flat = []
        for _, batch_y in data:
            for y_vector in batch_y:
                y_flat.append(y_vector)

        return y_flat, batch_predictions


class RegressionTask(Task, ABC):
    def __init__(self, base_model: BaseModel, task_type: TaskType, n_outputs=1, n_epochs=N_EPOCHS, load_weights=True,
                 load_on_init=True):
        super().__init__(base_model, task_type, n_outputs, n_epochs, load_weights, load_on_init)

    def eval(self):
        y_test, y_pred = self.get_predictions(self.validation_data())
        y_true = np.concatenate(y_test, axis=0)
        if self.is_regression:
            mae = mean_absolute_error(y_true, y_pred)
            print("Test Mean Absolute Error:", mae)


class ClassificationTask(Task, ABC):
    def __init__(self, base_model: BaseModel, task_type: TaskType, n_outputs=1, n_epochs=N_EPOCHS, load_weights=True,
                 load_on_init=True):
        super().__init__(base_model, task_type, n_outputs, n_epochs, load_weights, load_on_init)

    def eval(self, ):
        food2Index = Food2Index()
        y_test, y_pred = self.get_predictions(self.validation_data())  # no validation data on any class. task

        predictions = []
        labels = []

        class_tp = {}
        class_fp = {}
        for test_vector, pred_vector in zip(y_test, y_pred):
            pred = np.argmax(pred_vector)
            label = np.argmax(test_vector)

            predictions.append(pred)
            labels.append(label)

            if pred == label:
                name = food2Index.get_ingredient(label)
                if label in class_tp:
                    class_tp[name] += 1
                else:
                    class_tp[name] = 1
            else:
                name = food2Index.get_ingredient(pred)
                if pred in class_fp:
                    class_fp[name] += 1
                else:
                    class_fp[name] = 1

        print("eval", "-" * 25)
        print("Labels:\t", labels)
        print("Predictions", predictions)

        print("*" * 10, "TP", "*" * 10)
        pprint(class_tp)
        print("*" * 10, "FP", "*" * 10)
        pprint(class_fp)


def pprint(obj):
    print(json.dumps(obj, indent=4, sort_keys=True))
