from .Server_fed import FedServer

class ServerFedAvg(FedServer):
    def compute_step(self):        
        for name, _ in self.model.named_parameters():                        
            self.step[name] = self.pseudo_grad[name]