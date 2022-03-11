from .server_federated import FedServer
import torch

class ServerFedAdam(FedServer):
    def compute_step(self):      
        super(ServerFedAdam, self).compute_pseudo_gradient()  
        super(ServerFedAdam, self).update_m_vector()
        for name, _ in self.model.named_parameters():    
            self.v_vector[name] = self.server_momentum_param_2 * self.v_vector[name] + (1.0-self.server_momentum_param_2) * torch.square(self.pseudo_grad[name])
            self.step[name] = - torch.div( self.server_learning_rate * self.m_vector[name], torch.sqrt(self.v_vector[name]) + self.server_adapt_param )
