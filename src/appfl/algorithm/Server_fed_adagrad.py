from .Server_fed import FedAdaptServer
import torch

class FedAdagradServer(FedAdaptServer):
    def compute_step(self):        
        super(FedAdagradServer, self).update_m_vector()
        for name, _ in self.model.named_parameters():    
            self.v_vector[name] = self.v_vector[name] + torch.square(self.pseudo_grad[name])
            self.step[name] = torch.div( self.server_learning_rate * self.m_vector[name], torch.sqrt(self.v_vector[name]) + self.server_adapt_param )