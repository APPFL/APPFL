from .Server_fed import FedAdaptServer
import torch

class FedYogiServer(FedAdaptServer):
    def compute_step(self):        
        super(FedYogiServer, self).update_m_vector()
        for name, _ in self.model.named_parameters():    
            self.v_vector[name] = self.v_vector[name] - (1.0-self.server_momentum_param_2) * torch.mul( torch.square(self.pseudo_grad[name]), torch.sign(self.v_vector[name] - torch.square(self.pseudo_grad[name]) ) )            
            self.step[name] = torch.div( self.server_learning_rate * self.m_vector[name], torch.sqrt(self.v_vector[name]) + self.server_adapt_param )