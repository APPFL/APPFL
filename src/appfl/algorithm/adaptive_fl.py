import torch
from .fl_base import BaseServer, BaseClient


class adaptive_fl_Server(BaseServer):
    def __init__(self, weights, model, loss_fn, num_clients, device, server_lr=1.0):
        super().__init__(weights, model, loss_fn, num_clients, device)
        self.server_lr = server_lr

    def aggregate_gradients(self, gradients):
        """
        Aggregate gradients from all clients and update the global model.
        """
        total_gradient = sum(gradients) / self.num_clients
        with torch.no_grad():
            for param in self.model.parameters():
                param -= self.server_lr * total_gradient
    def run(self, max_rounds, clients):
        """
        Run the adaptive federated learning algorithm.
        
        Args:
            max_rounds (int): Number of communication rounds
            clients (list): List of AdaptiveFLClient objects
        """
        for round_idx in range(max_rounds):
            global_model = self.get_model()

            gradients = []
            for client in clients:
                gradient, func_val_diff = client.local_train(global_model)
                client.update_learning_rate(gradient, func_val_diff)
                gradients.append(gradient)

            # Aggregate gradients and update the global model
            self.aggregate_gradients(gradients)

            # Optionally log progress
            print(f"Round {round_idx+1}/{max_rounds} completed.")            



class adaptive_fl_client(BaseClient):
    def __init__(self, id, weight, model, loss_fn, dataloader, cfg, outfile, test_dataloader=None, metric=None, initial_lr=0.1, gamma=1.1):
        super().__init__(id, weight, model, loss_fn, dataloader, cfg, outfile, test_dataloader, metric)
        self.lr = initial_lr
        self.gamma = gamma

    def local_train(self, global_model):
        """
        Perform local training and return the gradient and function value difference.
        """
        self.model.load_state_dict(global_model)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        self.model.train()
        total_loss = 0.0
        for data, target in self.dataloader:
            data, target = data.to(self.cfg.device), target.to(self.cfg.device)
            optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        gradient = self._compute_gradient(global_model)
        func_val_diff = total_loss - self._compute_global_func_val(global_model)
        return gradient, func_val_diff

    def _compute_gradient(self, global_model):
        """
        Compute the gradient of the client's model compared to the global model.
        """
        gradient = []
        for param, global_param in zip(self.model.parameters(), global_model.values()):
            gradient.append(param.data - global_param)
        return torch.cat([g.view(-1) for g in gradient])

    def _compute_global_func_val(self, global_model):
        """
        Compute the objective function value for the global model.
        """
        total_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for data, target in self.dataloader:
                data, target = data.to(self.cfg.device), target.to(self.cfg.device)
                output = global_model(data)
                loss = self.loss_fn(output, target)
                total_loss += loss.item()
        return total_loss

    def update_learning_rate(self, gradient, func_val_diff):
        """
        Adaptive step search to update the learning rate.
        """
        if func_val_diff <= -self.lr * torch.norm(gradient) ** 2:
            # Successful step: increase learning rate
            self.lr *= self.gamma
        else:
            # Unsuccessful step: decrease learning rate
            self.lr /= self.gamma    





        