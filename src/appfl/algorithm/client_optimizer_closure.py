from .client_optimizer import ClientOptim

import torch
from torch.optim import *

class ClientOptimClosure(ClientOptim):
    
    def training(self):
        """
        This function trains the model using "optimizer" such as LBFGS which requires to reevaluate the function multiple times,
        """
        optimizer = eval(self.optim)(self.model.parameters(), **self.optim_args)

        train_loss = 0
        train_correct = 0
        tmptotal = 0

        for data, target in self.dataloader:

            tmptotal += len(target)

            data, target = data.to(self.cfg.device), target.to(self.cfg.device)

            def closure():
                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                output = self.model(data)

                # Compute loss
                loss = self.loss_fn(output, target)

                # Backward pass
                loss.backward()

                return loss

            optimizer.step(closure)

            loss = closure()
            train_loss += loss.data.item()
            output = self.model(data)
            train_correct = self.counting_correct(output, target, train_correct)

        train_loss = train_loss / len(self.dataloader)

        train_accuracy = 100.0 * train_correct / tmptotal

        return train_loss, train_accuracy

