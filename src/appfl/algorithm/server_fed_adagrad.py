from .server_federated import FedServer
import torch

class ServerFedAdagrad(FedServer):
    def compute_step(self):
        super(ServerFedAdagrad, self).compute_pseudo_gradient()        
        super(ServerFedAdagrad, self).update_m_vector()
        for name, _ in self.model.named_parameters():    
            self.v_vector[name] = self.v_vector[name] + torch.square(self.pseudo_grad[name])
            self.step[name] = - torch.div( self.server_learning_rate * self.m_vector[name], torch.sqrt(self.v_vector[name]) + self.server_adapt_param )