import os
import copy
import time
import torch
import numpy as np
from torch.optim import *
from .fl_base import BaseClient


class ALS(torch.optim.Optimizer):
    """Adaptive Line Search"""
    def __init__(self, params, alpha_0=1.0, alpha_max=10.0, inc_gamma=1.25, dec_gamma=0.7, theta=0.2, eps_f=0.0):
        if alpha_0 <= 0.0:
            raise ValueError("Invalid initial_step value: {}".format(alpha_0))
        if alpha_max <= 0.0 or alpha_max <= alpha_0:
            raise ValueError("Invalid maximum_step value: {}".format(alpha_max))
        if dec_gamma <= 0.0 or dec_gamma>=1:
            raise ValueError("Invalid decreasing_gamma value: {}".format(dec_gamma))
        if inc_gamma<=1:
            raise ValueError("Invalid increasing_gamma value: {}".format(inc_gamma))
        if theta <= 0.0 or theta >= 1:
            raise ValueError("Invalid theta value: {}".format(theta))

        defaults = dict(eps_f=eps_f, alpha_0=alpha_0, alpha_max=alpha_max, inc_gamma=inc_gamma, dec_gamma=dec_gamma, theta=theta)
        super(ALS, self).__init__(params, defaults)
        
    def step(self, closure):
        assert closure is not None, "A closure that reevaluates the model and returns the loss is required"

        loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                original_p = p.data.clone()
                alpha = group['alpha_0']
            
                #Tentative update
                p.data.add_(grad, alpha=-alpha)

                #Re-evaluate loss after tentative update
                new_loss = closure()

                #Line search condition
                if new_loss.item() <= loss.item() - alpha * group['theta'] * (grad.norm().item()**2) + 2*group['eps_f']:
                    #Sufficient decrease, increase step size
                    group['alpha_0'] = min(group['inc_gamma']*alpha, group['alpha_max'])
                else:
                    #Not sufficient decrease, revert update and decrease step size
                    p.data.copy_(original_p)
                    group['alpha_0'] *= group['dec_gamma']

        return loss  


class ClientAdaptiveOptim(BaseClient):
    """This client optimizer which perform updates for certain number of epochs in each training round."""
    def __init__(
        self, id, weight, model, loss_fn, dataloader, cfg, outfile, test_dataloader, metric, **kwargs
    ):
        super(ClientAdaptiveOptim, self).__init__(id, weight, model, loss_fn, dataloader, cfg, outfile, test_dataloader, metric)
        self.__dict__.update(kwargs)
        super(ClientAdaptiveOptim, self).client_log_title()

    def update(self):
        self.model.to(self.cfg.device)
        optimizer = ALS(self.model.parameters())
        
        ## Initial evaluation
        if self.cfg.validation == True and self.test_dataloader != None:
            start_time=time.time()
            test_loss, test_accuracy = super(ClientAdaptiveOptim, self).client_validation()
            per_iter_time = time.time() - start_time
            super(ClientAdaptiveOptim, self).client_log_content(0, per_iter_time, 0, 0, test_loss, test_accuracy)   

        num_epochs_per_estimate = 1

        #List of Individual data points in client to calculate noise(eps_f)
        x = []
        y = []
        for batch in self.dataloader:
            data_points, targets = batch[0], batch[1]
            x.append(data_points)
            y.append(targets)

        x = torch.cat(x, dim=0)
        y = torch.cat(y, dim=0) 

        ## Local training 
        for t in range(self.num_local_epochs):
            start_time=time.time()
            train_loss, target_true, target_pred = 0, [], []

            if (t % num_epochs_per_estimate==0):
                eps_f = self.estimate_epi_f(x, y, self.dataloader.batch_size)
                for group in optimizer.param_groups:
                    group['eps_f'] = eps_f

            for data, target in self.dataloader: 
                
                data = data.to(self.cfg.device)
                target = target.to(self.cfg.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_fn(output, target)
                
                """ Function to re-evaluate the loss after updating the parameter"""
                def closure():
                    output = self.model(data)
                    loss = self.loss_fn(output, target)
                    return loss
                
                loss.backward()
                optimizer.step(closure)

                target_true.append(target.detach().cpu().numpy())
                target_pred.append(output.detach().cpu().numpy())
                train_loss += loss.item() 
                if self.clip_grad or self.use_dp:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.clip_value,
                        norm_type=self.clip_norm,
                    )


            train_loss /= len(self.dataloader)
            target_true, target_pred = np.concatenate(target_true), np.concatenate(target_pred)
            train_accuracy = float(self.metric(target_true, target_pred))
            
            ## Validation
            if self.cfg.validation == True and self.test_dataloader != None:
                test_loss, test_accuracy = super(ClientAdaptiveOptim, self).client_validation()
            else:
                test_loss, test_accuracy = 0, 0
            per_iter_time = time.time() - start_time
            super(ClientAdaptiveOptim, self).client_log_content(t+1, per_iter_time, train_loss, train_accuracy, 0, 0)

            ## save model.state_dict()
            if self.cfg.save_model_state_dict == True:
                path = self.cfg.output_dirname + "/client_%s" % (self.id)
                if not os.path.exists(path):
                    os.makedirs(path, exist_ok=True)
                torch.save(self.model.state_dict(), os.path.join(path, "%s_%s.pt" % (self.round, t)))
 
        self.round += 1

        ## Differential Privacy
        self.primal_state = copy.deepcopy(self.model.state_dict())
        if self.use_dp:
            sensitivity = 2.0 * self.clip_value * self.optim_args.lr
            scale_value = sensitivity / self.epsilon
            super(ClientAdaptiveOptim, self).laplace_mechanism_output_perturb(scale_value)

        ## Move the model parameter to CPU (if not) for communication
        if (self.cfg.device == "cuda"):            
            for k in self.primal_state:
                self.primal_state[k] = self.primal_state[k].cpu()

        return self.primal_state
    
 
    def zeroth_oracle(self, x, y, sample_size):
        f_noisy_val = 0.0
        indices = torch.randperm(len(x))[:sample_size]

        for ind in indices:
            #Move to GPU
            data_point = x[ind].unsqueeze(0).to(self.cfg.device)
            target = y[ind].unsqueeze(0).to(self.cfg.device)

            output = self.model(data_point)
            f_noisy_val += self.loss_fn(output, target).item()

            #Move back to CPU and detach from the computational graph
            data_point = data_point.detach().cpu()
            target = target.detach().cpu()

        f_noisy_val /= sample_size
        return f_noisy_val
    
    
    def estimate_epi_f(self, x, y, sample_size, n_trials = 30, factor = 1/5):
        """Function to estimate noise"""
        result_arr = torch.zeros(n_trials, device=self.cfg.device)
        for i in range(n_trials):
            result_arr[i] = self.zeroth_oracle(x, y, sample_size)

        return torch.std(result_arr).item() * factor