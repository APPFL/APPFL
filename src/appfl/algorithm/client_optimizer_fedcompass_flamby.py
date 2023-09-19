import copy
import time
import torch
import numpy as np
from torch.optim import *
from collections import OrderedDict
from .algorithm import BaseClient

class ClientOptimFedCompassFlamby(BaseClient):
    def __init__(self, id, weight, model, loss_fn, dataloader, cfg, outfile, test_dataloader, metric, **kwargs):
        super(ClientOptimFedCompassFlamby, self).__init__(id, weight, model, loss_fn, dataloader, cfg, outfile, test_dataloader)
        super(ClientOptimFedCompassFlamby, self).client_log_title()
        self.__dict__.update(kwargs)
        self.round = 0
        self.metric = metric
    
    def client_validation(self, dataloader):
        if self.loss_fn is None or dataloader is None:
            return 0.0, 0.0
        self.model.to(self.cfg.device)
        self.model.eval()
        loss, tmpcnt = 0, 0
        with torch.no_grad():
            for img, target in dataloader:
                tmpcnt += 1
                img = img.to(self.cfg.device)
                target = target.to(self.cfg.device)
                output = self.model(img)
                loss += self.loss_fn(output, target).item()
        loss = loss / tmpcnt
        accuracy = self._evaluate_model_on_tests(dataloader)
        return loss, accuracy
    
    def update(self):
        self.model.to(self.cfg.device)
        optimizer = eval(self.optim)(self.model.parameters(), **self.optim_args)
        start_time=time.time()

        # initial evaluation
        if self.cfg.validation == True and self.test_dataloader != None:
            test_loss, metric_res = self.client_validation(self.test_dataloader)
            per_iter_time = time.time() - start_time
            super(ClientOptimFedCompassFlamby, self).client_log_content(0, per_iter_time, 0, 0, test_loss, metric_res)
            self.model.train()

        # local training
        data_iter = iter(self.dataloader)
        start_time = time.time()
        train_loss, tmptotal = 0, 0
        epoch = 1
        for _ in range(self.local_steps):
            try: 
                data, target = next(data_iter)
            except: # End of one local epoch
                ## Validation
                train_loss = train_loss / len(self.dataloader)
                if self.cfg.validation == True and self.test_dataloader != None:
                    test_loss, metric_res = self.client_validation(self.test_dataloader)
                    per_iter_time = time.time() - start_time
                    super(ClientOptimFedCompassFlamby, self).client_log_content(epoch, per_iter_time, train_loss, 0, test_loss, metric_res)
                    self.model.train()
                start_time = time.time()
                train_loss, tmptotal = 0, 0
                epoch += 1
                ## Reset the data iterator
                data_iter = iter(self.dataloader)
                data, target = next(data_iter)

            tmptotal += len(target)          
            data = data.to(self.cfg.device)
            target = target.to(self.cfg.device)
            optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if self.clip_value != False:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value, norm_type=self.clip_norm)

        self.round += 1
        self.primal_state = copy.deepcopy(self.model.to('cpu').state_dict())
        if (self.cfg.device == "cuda"):            
            for k in self.primal_state:
                self.primal_state[k] = self.primal_state[k].cpu()

        """ Differential Privacy  """
        if self.epsilon != False:
            sensitivity = 0
            if self.clip_value != False:
                sensitivity = 2.0 * self.clip_value * self.optim_args.lr
            scale_value = sensitivity / self.epsilon
            super(ClientOptimFedCompassFlamby, self).laplace_mechanism_output_perturb(scale_value)

        """ Update local_state """
        self.local_state = OrderedDict()
        self.local_state["primal"] = copy.deepcopy(self.primal_state)
        self.local_state["dual"] = OrderedDict()
        self.local_state["penalty"] = OrderedDict()
        self.local_state["penalty"][self.id] = 0.0
        
        return self.local_state
    
    def _evaluate_model_on_tests(self, test_dataloader):
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()
        with torch.no_grad():
            test_dataloader_iterator = iter(test_dataloader)
            y_pred_final = []
            y_true_final = []
            for (X, y) in test_dataloader_iterator:
                if torch.cuda.is_available():
                    X = X.cuda()
                    y = y.cuda()
                y_pred = self.model(X).detach().cpu()
                y = y.detach().cpu()
                y_pred_final.append(y_pred.numpy())
                y_true_final.append(y.numpy())

            y_true_final = np.concatenate(y_true_final)
            y_pred_final = np.concatenate(y_pred_final)
            accuracy = float(self.metric(y_true_final, y_pred_final))
        return accuracy
