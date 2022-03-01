from .fedadapt import FedAdaptServer

class FedAvgServer(FedAdaptServer):
    def compute_step(self):        
        for name, _ in self.model.named_parameters():                        
            self.step[name] = self.pseudo_grad[name]