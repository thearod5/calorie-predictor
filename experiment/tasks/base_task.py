import os
from abc import abstractmethod
from enum import Enum

import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from tensorflow.keras.metrics import mean_absolute_error

from constants import INPUT_SHAPE, N_EPOCHS, PROJECT_DIR
from experiment.tasks.test_model import test_model


def convert_to_task(base_model_class, n_outputs: int):
    inputs = tf.keras.Input(shape=INPUT_SHAPE)
    base_model = base_model_class(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs
    )
    added_dense_layer = tf.keras.layers.Dense(n_outputs)(base_model.layers[-2].output)
    model = tf.keras.Model(inputs=inputs, outputs=added_dense_layer)
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

    def load_model(self):
        if self.is_regression:
            self.model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        else:
            self.model.compile(optimizer="adam",
                               loss=tf.keras.losses.BinaryCrossentropy(),
                               metrics=["accuracy"])

        if self.load_weights and os.path.isdir(os.path.dirname(self.checkpoint_path)):
            print("Loading previous weights...")
            self.model.load_weights(self.checkpoint_path)
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

    def evaluate(self):
        self.model.summary()
        x_test = [x for x, _ in self.test_data()]
        y_test = [y for _, y in self.test_data()]
        y_pred = self.model.predict(x_test)
        y_true = np.concatenate(y_test, axis=0)
        if self.is_regression:
            mae = mean_absolute_error(y_true, y_pred)
            print("Test Mean Absolute Error:", mae)
        else:
            y_pred = list(map(lambda pred: np.argmax(pred), y_pred))
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            print("# True Negative: ", tn)
            print("# True Positive: ", tp)
            print("# False Positive: ", tn)
            print("# False Negative: ", tn)
