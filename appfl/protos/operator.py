import logging
from collections import OrderedDict
import hydra
from omegaconf import DictConfig

import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torchvision
from torchvision.transforms import ToTensor

from models import CNN
import numpy as np

from algorithm import utils
from protos.federated_learning_pb2 import Job

class FLOperator():
    def __init__(self, cfg):
        self.logger = logging.getLogger(__name__)
        self.operator_id = cfg.operator.id
        self.num_clients = cfg.num_clients
        self.num_epochs = cfg.num_epochs
        self.round_number = 1

        if cfg.validation == True:
            if cfg.dataset.torchvision == True:
                test_data = eval("torchvision.datasets." + cfg.dataset.classname)(
                    f"./datasets/0",
                    **cfg.dataset.args,
                    train=False,
                    transform=ToTensor(),
                )
                dataloader = DataLoader(test_data, batch_size=cfg.batch_size)
            else:
                raise NotImplementedError
        else:
            dataloader = None

        self.device = "cpu"
        self.dataloader = dataloader
        if self.dataloader is not None:
            self.loss_fn = CrossEntropyLoss()
        else:
            self.loss_fn = None

        self.model = eval(cfg.model.classname)(**cfg.model.args)
        self.validate_model = cfg.validation
        self.client_states = {}
        self.servicer = None

    def get_tensor(self, name):
        return np.array(self.model.state_dict()[name]) if name in self.model.state_dict() else None

    def get_job(self):
        job_todo = Job.TRAIN
        if self.round_number > self.num_epochs:
            job_todo = Job.QUIT
        return min(self.round_number, self.num_epochs), job_todo

    def update_weights(self):
        self.logger.info(f"[Round: {self.round_number: 04}] Updating model weights")

        aggr_state = {}
        for c in range(0,self.num_clients):
            for k,v in self.client_states[(c+1,self.round_number)].items():
                if k in aggr_state:
                    aggr_state[k] += v / self.num_clients
                else:
                    aggr_state[k] = v / self.num_clients

        new_state = {}
        for k in self.model.state_dict():
            new_state[k] = torch.from_numpy(aggr_state[k])
        self.model.load_state_dict(new_state)

        if self.validate_model:
            test_loss, accuracy = utils.validation(self.model, self.loss_fn, self.dataloader, self.device)
            self.logger.info(
                f"[Round: {self.round_number: 04}] Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%"
            )
        self.round_number += 1

    def is_round_finished(self):
        return all((c+1,self.round_number) in self.client_states for c in range(0,self.num_clients))

    def send_learning_results(self, client_id, round_number, tensor_list):
        results = {}
        for tensor in tensor_list:
            name = tensor.name
            shape = tuple(tensor.data_shape)
            flat = np.frombuffer(tensor.data_bytes, dtype=np.float32)
            nparray = np.reshape(flat, newshape=shape, order='C')
            results[name] = nparray
        self.client_states[(client_id,round_number)] = results

        if self.is_round_finished():
            self.logger.info(f"[Round: {self.round_number: 04}] Finished; all clients have sent their results.")
            self.update_weights()
