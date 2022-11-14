import json
import os

import tensorflow as tf
import torch
from tensorflow.python.keras.engine.keras_tensor import KerasTensor
from tqdm import tqdm

from constants import CAM_PATH, get_data_dir
from datasets.abstract_dataset import AbstractDataset
from experiment.cam.cam_dataset_converter import CamDatasetConverter
from experiment.models.managers.model_manager import ModelManager


class CamTrainerMetrics:
    def __init__(self, epoch: int, log_path: str):
        self.epoch = epoch
        self.log_path = log_path
        self.log = {'iterations': [], 'epoch': [], 'validation': [], 'train_acc': [], 'val_acc': []}
        self.train_step = 0
        self.accuracy = 0.
        self.total = 0
        self.c = 0
        self.tloss = 0.

    def get_train_loss(self):
        return self.tloss / self.c

    def get_accuracy(self):
        return self.accuracy / self.total

    def log_step_progress(self, loss):
        self.log['iterations'].append(loss.item())
        self.log['epoch'].append(self.tloss / self.c)
        self.log['train_acc'].append(self.accuracy / self.total)
        print('Epoch: ', self.epoch, 'Train loss: ', self.get_train_loss(), 'Accuracy: ', self.get_accuracy())

    def save_log(self):
        log_file_name = "epoch_log_%d.json" % self.epoch
        log_export_path = os.path.join(self.log_path, log_file_name)
        with open(log_export_path, 'w') as out:
            json.dump(self.log, out)
        print("Logged:", log_file_name)


class CamTrainer:
    def __init__(self, model_manager: ModelManager, alpha=0.5, lr=0.005, weight_decay=1e-6, momentum=0.9):
        self.model_manager = model_manager
        self.model = model_manager.get_model()
        self.log_path = os.path.join(get_data_dir(), "logs")
        self.base_cam_path = CAM_PATH
        self.solver = None
        self.criterion = tf.keras.losses.MeanSquaredError()
        self.criterion_hmap = tf.keras.losses.MeanSquaredError()
        self.alpha = alpha
        self.learning_rate = lr
        self.weight_decay = weight_decay
        self.momentum = momentum

    def get_solver(self):
        if self.solver is None:
            # missing weight decay
            self.solver = tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=self.momentum)
        return self.solver

    def train(self, training_data: AbstractDataset, n_epochs=3, ):
        cam_path = os.path.join(self.base_cam_path, training_data.dataset_path_creator.name)
        cam_dataset = CamDatasetConverter(training_data, cam_path).convert()

        for epoch in range(1, n_epochs + 1):
            metrics = CamTrainerMetrics(epoch, self.log_path)
            self.perform_cam_epoch(cam_dataset, metrics)

    def perform_cam_epoch(self, training_data, epoch_metrics: CamTrainerMetrics):
        solver = self.get_solver()
        train_loss = []
        with torch.set_grad_enabled(True):
            for batch_idx, (data, cals, hmap) in enumerate(tqdm(training_data)):
                outputs = self.model(data)

                # Prediction of accuracy

                # mse = tf.math.sqrt(((tf.cast(cals, tf.float32) - tf.cast(outputs, tf.float32)) ** 2))
                # epoch_metrics.accuracy += mse
                # epoch_metrics.total += data[0]
                # cams = []
                # for feature_index in range(n_features):
                #     feature_map = features[:, :, feature_index]
                #     feature_weight = feature_weights[feature_index]
                #     cam_img = feature_map * feature_weight
                #     cam_img = self.normalize(cam_img)
                #     cams.append(cam_img)
                # cams = tf.stack(cams)

                # 1. Get features  # TODO: Figure out why batch size is None
                features = self.model_manager.get_feature_layer(self.model).output
                feature_weights = self.model_manager.get_parameters()
                batch_size, feature_height, feature_width, n_features = features.shape
                features = tf.reshape(features, (feature_height, feature_width, n_features))

                # 2. Compute Sk model (weighted sum of the feature maps
                weighted_features = tf.linalg.matmul(features, feature_weights)
                weighted_features = self.normalize(weighted_features)
                weighted_features = tf.convert_to_tensor(weighted_features)

                # 3. Resize hmaps to feature size
                hmap = tf.image.resize(hmap, (feature_width, feature_height))

                # 3. Compute class loss and feature map loss
                class_loss = self.criterion(cals, outputs)
                hmap_loss = self.criterion_hmap(hmap, weighted_features)

                class_loss = tf.convert_to_tensor(class_loss)
                hmap_loss = tf.convert_to_tensor(hmap_loss)

                @tf.function
                def loss_fn(a: float):
                    alpha = tf.Variable(a, name="alpha")
                    alpha_negated = tf.Variable(1 - a, name='bias')
                    return tf.convert_to_tensor(alpha * class_loss + alpha_negated * hmap_loss)

                # loss = self.alpha * class_loss + (1 - self.alpha) * hmap_loss
                KerasTensor
                var_list_fn = lambda: self.model.trainable_weights
                opt_op = self.solver.minimize(loss_fn(self.alpha), var_list=var_list_fn)
                opt_op.run()

                # train_loss.append(epoch_metrics.tloss / epoch_metrics.c)
                # epoch_metrics.train_step += 1
                # epoch_metrics.tloss += loss.item()
                # epoch_metrics.c += 1
                # epoch_metrics.log_step_progress(loss)

        epoch_metrics.save_log()
        self.save_model(solver, epoch_metrics)

    def save_model(self, solver, metrics: CamTrainerMetrics):
        states = {
            'epoch': metrics.epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': solver.state_dict(),
        }
        torch.save(states, os.path.join(self.log_path, 'current_model.pth'))

    @staticmethod
    def normalize(tensor):
        return tf.divide(
            tf.subtract(
                tensor,
                tf.reduce_min(tensor)
            ),
            tf.subtract(
                tf.reduce_max(tensor),
                tf.reduce_min(tensor)
            )
        )
