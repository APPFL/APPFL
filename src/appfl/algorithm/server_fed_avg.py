from .server_federated import FedServer

class ServerFedAvg(FedServer):
    def compute_step(self):        
        super(ServerFedAvg, self).compute_pseudo_gradient()
        for name, _ in self.model.named_parameters():
            self.step[name] = - self.pseudo_grad[name]