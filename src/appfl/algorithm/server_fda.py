import logging
from collections import OrderedDict

from .server_federated import FedServer
from .algorithm import BaseClient

import torch
from torch.optim import *
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import copy
from .client_optimizer import ClientOptim
import numpy as np
import time
import os

class ServerFedGP(FedServer):
    def __init__(self, weights, model, loss_fn, num_clients, beta, clients_size, device, **kwargs):
        super(ServerFedGP, self).__init__(weights, model, loss_fn, num_clients, device, **kwargs)
        # initialization
        self.beta = beta # bias-variance trade-off factor
        # self.target  # target idx to know how to aggregate the gradients
        self.source_grads = OrderedDict()
        self.target_grad = OrderedDict()
        self.clients_size = clients_size

    def compute_step(self):
        super(ServerFedGP, self).compute_pseudo_gradient()
        for name in self.model.state_dict():
        # for name, _ in self.model.named_parameters():
            self.step[name] = -self.pseudo_grad[name]
    
    def compute_source_target_gradients(self):
        for idx in range(self.num_clients):
            if idx == self.target:
                for name in self.model.state_dict():
                    self.target_grad[name] = torch.zeros_like(self.model.state_dict()[name])
                    self.target_grad[name] = self.primal_states[idx][name] - self.global_state[name] 
            else:
                source_grad = OrderedDict()
                for name in self.model.state_dict():
                    source_grad[name] = torch.zeros_like(self.model.state_dict()[name])
                    source_grad[name] = self.primal_states[idx][name] - self.global_state[name] 
                self.source_grads[idx] = source_grad
    
    def update(self, local_states: OrderedDict, round_id):
        """Inputs for the global model update"""
        self.global_state = copy.deepcopy(self.model.state_dict())
        super(FedServer, self).primal_recover_from_local_states(local_states)

        """ residual calculation """
        super(FedServer, self).primal_residual_at_server()
 
        """ change device """
        for i in range(self.num_clients): 
            for name in self.model.state_dict():
                self.primal_states[i][name] = self.primal_states[i][name].to(
                    self.device
                )
        
        if round_id < 2:
            # go fedavg
            self.update_fedavg()
        
        else:
            # go fedgp
            self.update_fedgp()
    
    def update_fedavg(self):
        """ global_state calculation """
        self.compute_step() 
        for name in self.model.state_dict():   
            self.global_state[name] = (self.global_state[name]).float()     
            if name in self.list_named_parameters: 
                self.global_state[name] += self.step[name]            
            else:
                tmpsum = torch.zeros_like(self.global_state[name], device=self.device)                
                for i in range(self.num_clients):
                    tmpsum += self.primal_states[i][name]                
                self.global_state[name] = torch.div(tmpsum, self.num_clients)
                

        """ model update """
        self.model.load_state_dict(self.global_state)
    
    # update global model using the consine projection from target toward source directions
    def update_fedgp(self):
        self.compute_source_target_gradients()
        # ret_dict = copy.deepcopy(old_global_model_dict)
        b = self.beta
        cos = torch.nn.CosineSimilarity()
        for name in self.global_state:        
            # if name in self.list_named_parameters: 
            # if self.global_state[name].shape != torch.Size([]):
                # self.global_state[name] += self.step[name]
        # for key in ret_dict.keys():
            if self.global_state[name].shape != torch.Size([]):
                target_grad = self.target_grad[name] # target persudo gradient
                for idx in self.source_grads:
                    local_grad = self.source_grads[idx][name] # the rest of the persudo gradients
                    cur_sim = cos(target_grad.reshape(1,-1).float(), local_grad.reshape(1,-1).float())
                    if cur_sim > 0:
                        self.global_state[name] += b * self.target_lr_ratio * ((self.n_target_samples/self.target_batch_size)/(self.clients_size[idx]/self.source_batch_size)) * self.weights[idx] * cur_sim * local_grad
                self.global_state[name] += (1-b) * target_grad
            # else:
            #     tmpsum = torch.zeros_like(self.global_state[name], device=self.device)                
            #     for i in range(self.num_clients):
            #         tmpsum += self.primal_states[i][name]                
            #     self.global_state[name] = torch.div(tmpsum, self.num_clients)
        #         ret_dict[key] = old_global_model_dict[key]
        # return ret_dict
        
        """ model update """
        self.model.load_state_dict(self.global_state)
        
        
    def logging_summary(self, cfg, logger):
        super(FedServer, self).log_summary(cfg, logger)

        logger.info("client_learning_rate = %s " % (cfg.fed.args.optim_args.lr))

        if cfg.summary_file != "":
            with open(cfg.summary_file, "a") as f:

                f.write(
                    cfg.logginginfo.DataSet_name
                    + " Federated Gradient Projection ClientLR "
                    + str(cfg.fed.args.optim_args.lr)
                    + " TestAccuracy "
                    + str(cfg.logginginfo.accuracy)
                    + " BestAccuracy "
                    + str(cfg.logginginfo.BestAccuracy)
                    + " Time "
                    + str(round(cfg.logginginfo.Elapsed_time, 2))
                    + "\n"
                )

class FedMTLClient(BaseClient):
    def __init__(self, id, weight, model, loss_fn, dataloader, cfg, outfile, test_dataloader, **kwargs):
        super(FedMTLClient, self).__init__(
            id, weight, model, loss_fn, dataloader, cfg, outfile, test_dataloader
        )
        self.__dict__.update(kwargs)

        self.round = 0

        super(FedMTLClient, self).client_log_title()
    
    def client_validation_MTL(self, dataloader):
        
        # print('ewotuwt')

        if self.loss_fn is None or dataloader is None:
            return 0.0, 0.0

        self.model.to(self.cfg.device)
        self.model.eval()
        loss = 0
        correct = 0
        tmpcnt = 0
        tmptotal = 0
        with torch.no_grad():
            for sample in self.dataloader:          
                tmpcnt += 1      
                data = sample['img'].to(self.cfg.device)
                targets = sample['targets']
                target = targets[0].unsqueeze(1).to(self.cfg.device)
                tmptotal += len(target)               
                preds_all = self.model(data)
                output = preds_all[0]
                
                target = target.type_as(output)
                probs = torch.sigmoid(output)
                pred = probs > 0.5
                
                loss += self.loss_fn[0](probs, target).item()

                # if output.shape[1] == 1:
                #     pred = torch.round(output)
                # else:
                #     pred = output.argmax(dim=1, keepdim=True)

                correct += pred.eq(target.view_as(pred)).sum().item()

        # FIXME: do we need to sent the model to cpu again?
        # self.model.to("cpu")

        loss = loss / tmpcnt
        accuracy = 100.0 * correct / tmptotal

        return loss, accuracy
        
    # update with multiple losses
    def update(self):
        """Inputs for the local model update"""

        self.model.to(self.cfg.device)

        optimizer = eval(self.optim)(self.model.parameters(), **self.optim_args)
 

        """ Multiple local update """
        start_time=time.time()
        ## initial evaluation
        if self.cfg.validation == True and self.test_dataloader != None:
            test_loss, test_accuracy = self.client_validation_MTL(
                self.test_dataloader
            )
            per_iter_time = time.time() - start_time
            super(FedMTLClient, self).client_log_content(
                0, per_iter_time, 0, 0, test_loss, test_accuracy
            )
            ## return to train mode
            self.model.train()        

        ## local training 
        for t in range(self.num_local_epochs):
            start_time=time.time()
            train_loss = 0
            train_correct = 0            
            tmptotal = 0
            for sample in self.dataloader:                
                data = sample['img'].to(self.cfg.device)
                targets = sample['targets']
                
                for i in range(len(targets)):
                    targets[i] = targets[i].to(self.cfg.device)
                labels = targets[0].unsqueeze(1)
                tmptotal += len(labels)
                
                optimizer.zero_grad()
                
                preds_all = self.model(data)
                output = preds_all[0]
                
                labels = labels.type_as(output)
                probs = torch.sigmoid(output)
                pred = probs > 0.5
                    
                # TODO: setup multiple loss_fn and also test whether it is target client                
                # if self.id != self.cfg.fed.target:
                loss = self.loss_fn[0](probs, labels)
                for idx, c in enumerate(self.loss_fn[1:]):
                    loss += c(preds_all[idx+1], targets[idx+1])
                # else:
                    # loss = self.loss_fn[0](output, labels)

                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                # if output.shape[1] == 1:
                #     pred = torch.round(output)
                # else:
                #     pred = output.argmax(dim=1, keepdim=True)
                train_correct += pred.eq(labels.view_as(pred)).sum().item()

                if self.clip_value != False:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.clip_value,
                        norm_type=self.clip_norm,
                    )
            ## Validation
            train_loss = train_loss / len(self.dataloader)
            train_accuracy = 100.0 * train_correct / tmptotal
            if self.cfg.validation == True and self.test_dataloader != None:
                test_loss, test_accuracy = self.client_validation_MTL(
                    self.test_dataloader
                )
                per_iter_time = time.time() - start_time
                super(FedMTLClient, self).client_log_content(
                    t+1, per_iter_time, train_loss, train_accuracy, test_loss, test_accuracy
                )
                ## return to train mode
                self.model.train()

            ## save model.state_dict()
            if self.cfg.save_model_state_dict == True:
                path = self.cfg.output_dirname + "/client_%s" % (self.id)
                if not os.path.exists(path):
                    os.makedirs(path)
                torch.save(
                    self.model.state_dict(),
                    os.path.join(path, "%s_%s.pt" % (self.round, t)),
                )
 
        self.round += 1

        self.primal_state = copy.deepcopy(self.model.state_dict())
        if (self.cfg.device == "cuda"):            
            for k in self.primal_state:
                self.primal_state[k] = self.primal_state[k].cpu()

        """ Differential Privacy  """
        if self.epsilon != False:
            sensitivity = 0
            if self.clip_value != False:
                sensitivity = 2.0 * self.clip_value * self.optim_args.lr
            scale_value = sensitivity / self.epsilon
            super(FedMTLClient, self).laplace_mechanism_output_perturb(scale_value)

        """ Update local_state """
        self.local_state = OrderedDict()
        self.local_state["primal"] = self.primal_state
        self.local_state["dual"] = OrderedDict()
        self.local_state["penalty"] = OrderedDict()
        self.local_state["penalty"][self.id] = 0.0

        return self.local_state