import json
import os

import torch
from torch import nn, optim
from tqdm import tqdm

from constants import PATH_TO_OUTPUT_DIR
from datasets.abstract_dataset import AbstractDataset
from experiment.cam.cam_dataset_converter import CamDatasetConverter
from experiment.models.managers.model_manager import ModelManager


class CamTrainer:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.model = model_manager.get_model()
        self.log_path = os.path.join(PATH_TO_OUTPUT_DIR, "logs")
        self.base_cam_path = os.path.join(PATH_TO_OUTPUT_DIR, "cam")

    def train(self, training_data: AbstractDataset, n_epochs=3, alpha=1, lr=0.005, weight_decay=1e-6, momentum=0.9):
        cam_path = os.path.join(self.base_cam_path, training_data.__class__.__name__)
        cam_dataset = CamDatasetConverter(training_data, cam_path).convert()
        log = {'iterations': [], 'epoch': [], 'validation': [], 'train_acc': [], 'val_acc': []}
        criterion = nn.CrossEntropyLoss()
        criterion_hmap = nn.MSELoss()
        solver = optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)

        for epoch in range(1, n_epochs + 1):
            self.perform_cam_epoch(cam_dataset, criterion, criterion_hmap, alpha, epoch, log, solver)

    def perform_cam_epoch(self, training_data, criterion, criterion_hmap, alpha, epoch, log, solver):
        train_step = 0
        acc = 0.
        tot = 0
        c = 0
        tloss = 0.
        train_loss = []
        with torch.set_grad_enabled(True):
            for batch_idx, (data, cls, hmap) in enumerate(tqdm(training_data)):
                outputs = self.model(data)

                # Prediction of accuracy
                pred = torch.max(outputs, dim=1)[1]
                corr = torch.sum((pred == cls).int())
                acc += corr.item()
                tot += data.size(0)
                class_loss = criterion(outputs, cls)
            # Running model over data
            if alpha != 1:
                features = self.model_manager._get_feature_layer(self.model).output
                params = self.model_manager.get_parameters()

                bz, nc, h, w = features.shape

                beforeDot = features.reshape((bz, nc, h * w))
                cams = []
                for ids, bd in enumerate(beforeDot):
                    weight = params[pred[ids]]
                    cam = torch.matmul(weight, bd)
                    cam_img = cam.reshape(h, w)
                    cam_img = cam_img - torch.min(cam_img)
                    cam_img = cam_img / torch.max(cam_img)
                    cams.append(cam_img)
                cams = torch.stack(cams)
                hmap_loss = criterion_hmap(cams, hmap)
            else:
                hmap_loss = 0
            loss = alpha * class_loss + (1 - alpha) * hmap_loss if alpha != 1.0 else class_loss

            solver.zero_grad()
            loss.backward()
            solver.step()
            train_loss.append(tloss / c)
            train_step += 1
            tloss += loss.item()
            c += 1

            self.log_step_progress(acc, c, epoch, log, loss, tloss, tot)

        self.save_model(log, epoch, solver)

    def save_model(self, log, epoch, solver):
        states = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': solver.state_dict(),
        }
        log_index = len(os.listdir(self.log_path))
        log_file_name = "model_log_%s.json" % (log_index)
        with open(os.path.join(self.log_path, log_file_name), 'w') as out:
            json.dump(log, out)
        torch.save(states, os.path.join(self.log_path, 'current_model.pth'))

    @staticmethod
    def log_step_progress(acc, c, epoch, log, loss, tloss, tot):
        log['iterations'].append(loss.item())
        log['epoch'].append(tloss / c)
        log['train_acc'].append(acc / tot)
        print('Epoch: ', epoch, 'Train loss: ', tloss / c, 'Accuracy: ', acc / tot)
