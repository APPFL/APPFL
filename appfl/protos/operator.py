import logging
from collections import OrderedDict
import hydra
from omegaconf import DictConfig

import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torchvision
from torchvision.transforms import ToTensor

import numpy as np
import copy

from appfl.misc.utils import *
from appfl.algorithm.iadmm import *
from appfl.algorithm.fedavg import *
from .federated_learning_pb2 import Job

class FLOperator():
    def __init__(self, cfg, model, test_dataset, num_clients):
        self.logger = logging.getLogger(__name__)
        self.operator_id = cfg.operator.id
        self.cfg = cfg
        self.num_clients = num_clients
        self.num_epochs = cfg.num_epochs
        self.round_number = 1
        self.best_accuracy = 0.0
        self.device = "cpu"
        self.client_states = {}
        self.client_learning_status = {}
        self.servicer = None # Takes care of communication via gRPC

        self.dataloader = DataLoader(test_dataset,
                                     num_workers=0,
                                     batch_size=cfg.test_data_batch_size,
                                     shuffle=cfg.test_data_shuffle)
        self.fed_server = eval(cfg.fed.servername)(
            copy.deepcopy(model), num_clients, self.device, **cfg.fed.args)

    def get_tensor(self, name):
        return np.array(self.fed_server.model.state_dict()[name]) if name in self.fed_server.model.state_dict() else None

    def get_job(self):
        job_todo = Job.TRAIN
        if self.round_number > self.num_epochs:
            job_todo = Job.QUIT
        return min(self.round_number, self.num_epochs), job_todo

    def update_weights(self):
        self.logger.info(f"[Round: {self.round_number: 04}] Updating model weights")
        self.fed_server.update(self.fed_server.model.state_dict(), self.client_states)

        if self.cfg.validation == True:
            test_loss, accuracy = validation(self.fed_server, self.dataloader)

            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy

            self.logger.info(
                f"[Round: {self.round_number: 04}] Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%, Best Accuracy: {self.best_accuracy:.2f}%"
            )
        self.round_number += 1

    def is_round_finished(self):
        return all((c+1,self.round_number) in self.client_learning_status for c in range(0,self.num_clients))

    def send_learning_results(self, client_id, round_number, tensor_list):
        results = {}
        for tensor in tensor_list:
            name = tensor.name
            shape = tuple(tensor.data_shape)
            flat = np.frombuffer(tensor.data_bytes, dtype=np.float32)
            nparray = np.reshape(flat, newshape=shape, order='C')
            results[name] = torch.from_numpy(nparray)
        self.client_states[client_id] = results
        self.client_learning_status[(client_id,round_number)] = True

        if self.is_round_finished():
            self.logger.info(f"[Round: {self.round_number: 04}] Finished; all clients have sent their results.")
            self.update_weights()
