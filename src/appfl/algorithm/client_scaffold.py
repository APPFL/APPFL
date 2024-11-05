import copy
import torch
from collections import OrderedDict
from .fl_base import BaseClient

class SCAFFOLDClient(BaseClient):
    def __init__(self, id, weight, model, loss_fn, dataloader, cfg, outfile, test_dataloader, metric, local_lr=1e-2, **kwargs):
        super(SCAFFOLDClient, self).__init__(id, weight, model, loss_fn, dataloader, cfg, outfile, test_dataloader, metric)
        self.local_lr = local_lr
        self.local_control = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}  # Initialize client control variate
        self.__dict__.update(kwargs)
        
        if hasattr(self, 'outfile') and self.outfile:
            self.client_log_title()

    def client_log_title(self):
        title = "%10s %10s %10s %10s %10s %10s %10s \n" % (
            "Round",
            "LocalEpoch",
            "PerIter[s]",
            "TrainLoss",
            "TrainAccu",
            "TestLoss",
            "TestAccu",
        )
        self.outfile.write(title)
        self.outfile.flush()

    def update(self, global_model, global_control):
        # Load the global model and control variate
        self.model.load_state_dict(global_model)
        self.global_control = global_control
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.local_lr)

        # Initialize local model and perform local updates
        initial_model_state = copy.deepcopy(self.model.state_dict())
        
        for _ in range(self.num_local_epochs):
            for data, target in self.dataloader:
                data, target = data.to(self.cfg.device), target.to(self.cfg.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_fn(output, target)
                loss.backward()

                # Adjust gradients with control variates
                for param, (name, control) in zip(self.model.parameters(), self.local_control.items()):
                    if param.grad is not None:
                        param.grad -= (self.local_control[name] - self.global_control[name])

                optimizer.step()

        # Calculate delta_y and delta_c for the client
        delta_y = OrderedDict()
        delta_c = OrderedDict()
        for name, param in self.model.state_dict().items():
            delta_y[name] = param - initial_model_state[name]
            delta_c[name] = self.local_control[name] - self.global_control[name] + (1 / (self.num_local_epochs * self.local_lr)) * (global_model[name] - param)

        # Update local control variate
        self.local_control = {k: self.local_control[k] + delta_c[k] for k in delta_c}

        return {
            "delta_y": delta_y,
            "delta_c": delta_c
        }
