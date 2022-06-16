from .server_async_federated import AsyncFedServer
from .server_federated import FedServer

class ServerFedAsync(AsyncFedServer):
    def compute_step(self):
        super(ServerFedAsync, self).compute_pseudo_gradient()
        for name, _ in self.model.named_parameters():
            self.step[name] = -self.pseudo_grad[name]

    def logging_summary(self, cfg, logger):
        super(FedServer, self).log_summary(cfg, logger)